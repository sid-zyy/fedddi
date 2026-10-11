from __future__ import annotations
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch
NDArrays = List[np.ndarray]

def compute_block_index_map(encoder: Any, predictor: Any) -> Dict[str, List[int]]:
    """Map feature projection, convolution and predictor parameters to their wire-format indices."""
    blocks: Dict[str, List[int]] = {'feature_proj': [], 'convs': [], 'mlp': []}
    idx = 0
    for name, _ in encoder.named_parameters():
        if 'feature_proj' in name:
            blocks['feature_proj'].append(idx)
            idx += 1
        elif 'convs' in name:
            blocks['convs'].append(idx)
            idx += 1
        else:
            raise ValueError(f"Unrecognised encoder parameter {name!r} — CAGA's block map must account for every aggregated parameter. If the encoder architecture changed, update compute_block_index_map.")
    for name, _ in predictor.named_parameters():
        if 'mlp' in name:
            blocks['mlp'].append(idx)
            idx += 1
        else:
            raise ValueError(f"Unrecognised predictor parameter {name!r} — CAGA's block map must account for every aggregated parameter.")
    return blocks

def sample_probe_edges(
    root_pos_edges: torch.Tensor,
    num_nodes: int,
    m: int,
    seed: int,
    device: torch.device,
    exclude_edge_index: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sample root positives and random negatives. Negatives avoid `exclude_edge_index` (pass the
    full public graph so no known interaction is used as a negative); without it they avoid only the
    root edges. The seed controls both samples and the global RNG state is left untouched.
    """
    from torch_geometric.utils import negative_sampling
    g = torch.Generator().manual_seed(seed)
    n_pos = m // 2
    n_pool = root_pos_edges.size(0)
    if n_pool == 0:
        raise ValueError('Root pool is empty — cannot sample probe edges.')
    if n_pool >= n_pos:
        perm = torch.randperm(n_pool, generator=g)[:n_pos]
    else:
        perm = torch.randint(0, n_pool, (n_pos,), generator=g)
    pos_edges = root_pos_edges[perm]
    device = torch.device(device)
    if exclude_edge_index is not None:
        edge_index_for_neg = exclude_edge_index.to(device)
    else:
        edge_index_for_neg = root_pos_edges.t().to(device)
    py_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=[device] if device.type == 'cuda' else []):
            torch.manual_seed(seed)
            random.seed(seed)
            neg_pairs = negative_sampling(
                edge_index=edge_index_for_neg,
                num_nodes=num_nodes,
                num_neg_samples=m - n_pos
            ).t()
    finally:
        random.setstate(py_state)
    probe = torch.cat([pos_edges.to(device), neg_pairs.to(device)], dim=0)
    return probe

def _set_full_parameters(
    encoder: Any,
    predictor: Any,
    block_map: Dict[str, List[int]],
    arrays: NDArrays
) -> None:
    """Load candidate arrays into the scratch model using the block index map."""
    name_order = [name for name, _ in encoder.named_parameters()]
    state_dict = encoder.state_dict()
    for name in name_order:
        if 'feature_proj' in name:
            pos = block_map['feature_proj'][[n for n in name_order if 'feature_proj' in n].index(name)]
        elif 'convs' in name:
            pos = block_map['convs'][[n for n in name_order if 'convs' in n].index(name)]
        else:
            continue
        state_dict[name] = torch.tensor(arrays[pos], dtype=torch.float32)
    encoder.load_state_dict(state_dict, strict=False)
    pred_names = [name for name, _ in predictor.named_parameters() if 'mlp' in name]
    state_dict_pred = predictor.state_dict()
    for local_i, name in enumerate(pred_names):
        pos = block_map['mlp'][local_i]
        state_dict_pred[name] = torch.tensor(arrays[pos], dtype=torch.float32)
    predictor.load_state_dict(state_dict_pred, strict=False)

def score_arrays_on_probe(
    root_trainer: Any,
    block_map: Dict[str, List[int]],
    arrays: NDArrays,
    probe_edges: torch.Tensor
) -> np.ndarray:
    """Return candidate logits using clean server features and the shared graph."""
    from src.training.evaluate import score_edges
    _set_full_parameters(root_trainer.encoder, root_trainer.predictor, block_map, arrays)
    root_trainer.encoder.eval()
    root_trainer.predictor.eval()
    with torch.no_grad():
        h = root_trainer.encoder(root_trainer.edge_index)
        logits = score_edges(
            h,
            root_trainer.predictor,
            probe_edges,
            batch_size=root_trainer.cfg.training.batch_size,
            device=root_trainer.device,
            node_dist_enc=root_trainer.dist_enc
        )
    return logits.cpu().numpy()
