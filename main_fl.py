"""Federated DDI training with CAGA, FedAvg, FedProx, norm-clipping and FLTrust baselines."""
from __future__ import annotations
import csv
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import hydra
import numpy as np
import torch
import wandb
from flwr.client import Client, ClientApp
from flwr.common import Context, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from omegaconf import DictConfig, OmegaConf
from src.data.dataset import get_edge_split, load_dataset
from src.data.poisoning import PoisonEvent, PoisoningSimulator, build_poison_events_from_fraction
from src.data.data_drift import FeatureDriftEvent, DataDriftSimulator
from src.data.local_observations import generate_client_observations
from src.training.robust_aggregation import clip_and_average, compute_update_norm, median_norm_clip_threshold
from src.data.features import compute_distance_encoding, compute_fixed_global_features
from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.evaluate import evaluate, score_edge_set
from src.training.fl_client import FedDDIClient, _is_aggregated_encoder_param, _is_aggregated_predictor_param
from src.training.trainer import Trainer, build_client_trainer
from src.utils.utils import get_device, seed_everything, setup_logging
logger = logging.getLogger(__name__)

def build_server_evaluate_fn(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    full_edge_index: torch.Tensor,
    split: Dict[str, Any],
    evaluator: Any,
    device: torch.device,
    batch_size: int,
    node_dist_enc: torch.Tensor | None=None,
    using_shared_node_features: bool=False,
    on_round_eval: Callable[[int, float, float], None] | None=None
) -> Callable[[int, List[np.ndarray], Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None]:
    """Build the server-side evaluation function"""

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Tuple[float, Dict[str, Scalar]] | None:
        idx = 0
        enc_sd = encoder.state_dict()
        for name, _ in encoder.named_parameters():
            if _is_aggregated_encoder_param(name):
                enc_sd[name] = torch.tensor(parameters[idx], dtype=torch.float32).to(device)
                idx += 1
        encoder.load_state_dict(enc_sd, strict=False)
        pred_sd = predictor.state_dict()
        for name, _ in predictor.named_parameters():
            if _is_aggregated_predictor_param(name):
                pred_sd[name] = torch.tensor(parameters[idx], dtype=torch.float32).to(device)
                idx += 1
        predictor.load_state_dict(pred_sd, strict=False)
        val_hits = evaluate(
            encoder,
            predictor,
            full_edge_index,
            split['valid']['edge'],
            split['valid']['edge_neg'],
            evaluator,
            batch_size,
            device,
            node_dist_enc
        )
        test_hits = evaluate(
            encoder,
            predictor,
            full_edge_index,
            split['test']['edge'],
            split['test']['edge_neg'],
            evaluator,
            batch_size,
            device,
            node_dist_enc
        )
        h20_val = float(val_hits.get('Hits@20', 0.0))
        h20_test = float(test_hits.get('Hits@20', 0.0))
        if using_shared_node_features:
            tag = 'server_eval'
            label = ''
        else:
            tag = 'server_diag'
            label = ' [diagnostic-only — feature_proj not aggregated in this mode]'
        logger.info(
            'Round %d — %s Val Hits@20: %.4f | Test Hits@20: %.4f%s',
            server_round,
            tag,
            h20_val,
            h20_test,
            label
        )
        wandb.log({'round': server_round, f'{tag}_val_hits20': h20_val, f'{tag}_test_hits20': h20_test})
        if on_round_eval is not None:
            on_round_eval(server_round, h20_val, h20_test)
        return (0.0, {f'{tag}_Hits@20': h20_val})
    return evaluate_fn

def make_metrics_aggregation_fn(
    kind: str,
    best_tracker: Dict[str, float] | None=None,
    on_round: Callable[[int, Dict[str, float]], None] | None=None
) -> Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]:
    """Weighted-average aggregation for fit or evaluate metrics. Each metric is averaged only over
    the clients that reported it, so drift- and poison-specific metrics are not diluted by clients
    without drift or poisoning.
    """
    round_counter = {'n': 0}

    def aggregation_fn(results: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        if not results:
            return {}
        total_examples = sum((n for n, _ in results))
        if total_examples == 0:
            return {}
        aggregated: Dict[str, float] = {}
        weights: Dict[str, int] = {}
        for num_examples, metrics in results:
            for key, value in metrics.items():
                if key == 'cid':
                    continue
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples
                    weights[key] = weights.get(key, 0) + num_examples
        aggregated = {key: total / weights[key] for key, total in aggregated.items() if weights[key] > 0}
        round_counter['n'] += 1
        rnd = round_counter['n']
        if kind == 'evaluate':
            val_h20 = aggregated.get('Hits@20', 0.0)
            logger.info('Round %d — federated Val Hits@20: %.4f', rnd, val_h20)
            wandb.log({'round': rnd, 'federated_val_hits20': val_h20})
            if best_tracker is not None and val_h20 > best_tracker.get('val_hits20', -1.0):
                best_tracker['val_hits20'] = val_h20
                best_tracker['round'] = float(rnd)
        elif kind == 'fit':
            loss = aggregated.get('loss', float('nan'))
            logger.info('Round %d — federated train loss: %.4f', rnd, loss)
            wandb.log({'round': rnd, 'federated_train_loss': loss})
        if on_round is not None:
            on_round(rnd, aggregated)
        return cast(Dict[str, Scalar], aggregated)
    return aggregation_fn

class CAGAStrategy(FedAvg):
    """Flower strategy supporting FedAvg and FedProx"""

    def __init__(
        self,
        *,
        algorithm: str,
        num_total_clients: int,
        defense_mode: str='none',
        clip_multiplier: float=1.5,
        on_defense_stats: Optional[Callable[[int, float, int, List[float]], None]]=None,
        root_trainer: Optional[Any]=None,
        caga_block_map: Optional[Dict[str, List[int]]]=None,
        caga_hp: Optional[Any]=None,
        on_caga_stats: Optional[Callable[[int, Any], None]]=None,
        server_update: Optional[Callable[[int, List[np.ndarray]], List[np.ndarray]]]=None,
        on_fltrust_stats: Optional[Callable[[int, Any], None]]=None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._algorithm = algorithm.lower()
        self._num_total_clients = num_total_clients
        if defense_mode not in ('none', 'norm_clip', 'caga', 'fltrust', 'root_client'):
            raise ValueError(f"Unknown defense_mode {defense_mode!r} — expected 'none', 'norm_clip', 'caga', 'fltrust' or 'root_client'.")
        if defense_mode in ('fltrust', 'root_client') and server_update is None:
            raise ValueError(f"defense_mode={defense_mode!r} requires server_update.")
        self._server_update = server_update
        self._on_fltrust_stats = on_fltrust_stats
        self._defense_mode = defense_mode
        self._clip_multiplier = clip_multiplier
        self._on_defense_stats = on_defense_stats
        self._last_parameters_ndarrays: Optional[List[Any]] = None
        self._root_trainer = root_trainer
        self._caga_block_map = caga_block_map
        if defense_mode == 'caga':
            if root_trainer is None or caga_block_map is None:
                raise ValueError("defense_mode='caga' requires root_trainer and caga_block_map.")
        from src.training.caga import CAGAHyperparams
        self._caga_hp = caga_hp if caga_hp is not None else CAGAHyperparams()
        self._on_caga_stats = on_caga_stats

    def configure_fit(self, server_round, parameters, client_manager):
        """Propagate the current round to each client before training."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        updated: List = []
        for client_proxy, fit_ins in fit_ins_list:
            from flwr.common import FitIns
            new_config = dict(fit_ins.config)
            new_config['server_round'] = server_round
            client_parameters = fit_ins.parameters
            updated.append((client_proxy, FitIns(client_parameters, new_config)))
        return updated

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Inject server_round into each client's evaluate config too (see configure_fit above —
        evaluate() uses the same _refresh_round_state mechanism in fl_client.py, so it needs the
        same injection).
        """
        eval_ins_list = super().configure_evaluate(server_round, parameters, client_manager)
        updated: List = []
        for client_proxy, eval_ins in eval_ins_list:
            from flwr.common import EvaluateIns
            new_config = dict(eval_ins.config)
            new_config['server_round'] = server_round
            updated.append((client_proxy, EvaluateIns(eval_ins.parameters, new_config)))
        return updated

    def initialize_parameters(self, client_manager):
        """Cache the initial global parameters — needed as the "old" reference point for computing
        client update norms on round 1's aggregate_fit, when defense_mode == "norm_clip".
        """
        parameters = super().initialize_parameters(client_manager)
        if parameters is not None:
            self._last_parameters_ndarrays = parameters_to_ndarrays(parameters)
        return parameters

    def aggregate_fit(self, server_round, results, failures):
        # Keep diagnostic arrays aligned with client partition IDs.
        results = sorted(results, key=lambda pair: int(pair[1].metrics.get('cid', 0)))
        if self._defense_mode == 'root_client':
            if not results:
                return (None, {})
            if self._last_parameters_ndarrays is None:
                self._last_parameters_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
            # Control for the server's clean data: its root update joins plain averaging as one more client.
            arrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            weights = [float(fit_res.num_examples) for _, fit_res in results]
            arrays.append(self._server_update(server_round, self._last_parameters_ndarrays))
            weights.append(float(np.mean(weights)))
            total = sum(weights)
            averaged = [sum(w * a[k] for w, a in zip(weights, arrays)) / total for k in range(len(arrays[0]))]
            self._last_parameters_ndarrays = averaged
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics) if self.fit_metrics_aggregation_fn else {}
            return (ndarrays_to_parameters(averaged), metrics_aggregated)
        if self._defense_mode == 'fltrust':
            if not results:
                return (None, {})
            if self._last_parameters_ndarrays is None:
                self._last_parameters_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
            from src.training.fltrust import fltrust_aggregate
            client_keys = [
                str(int(fit_res.metrics.get('cid', i))) for i,
                (_, fit_res) in enumerate(results)
            ]
            client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            # The server update starts from the same global model the clients received.
            server_ndarrays = self._server_update(server_round, self._last_parameters_ndarrays)
            averaged, fltrust_report = fltrust_aggregate(
                old_arrays=self._last_parameters_ndarrays,
                client_arrays=client_ndarrays,
                server_arrays=server_ndarrays,
                client_keys=client_keys,
                server_round=server_round
            )
            self._last_parameters_ndarrays = averaged
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics) if self.fit_metrics_aggregation_fn else {}
            if self._on_fltrust_stats is not None:
                self._on_fltrust_stats(server_round, fltrust_report)
            return (ndarrays_to_parameters(averaged), metrics_aggregated)
        if self._defense_mode == 'caga':
            if not results:
                return (None, {})
            if self._last_parameters_ndarrays is None:
                self._last_parameters_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
            from src.training.caga import caga_round, make_probe_loss_fn
            from src.training.caga_probe import sample_probe_edges
            client_keys = [
                str(int(fit_res.metrics.get('cid', i))) for i,
                (_, fit_res) in enumerate(results)
            ]
            client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            client_weights = [float(fit_res.num_examples) for _, fit_res in results]
            probe_edges = sample_probe_edges(
                root_pos_edges=self._root_trainer.split['train']['edge'],
                num_nodes=self._root_trainer.encoder.num_nodes,
                m=self._caga_hp.probe_size,
                seed=server_round,
                device=self._root_trainer.device,
                exclude_edge_index=self._root_trainer.edge_index
            )
            loss_fn = make_probe_loss_fn(self._root_trainer, self._caga_block_map, probe_edges)
            averaged, caga_report = caga_round(
                old_arrays=self._last_parameters_ndarrays,
                client_arrays=client_ndarrays,
                client_keys=client_keys,
                client_weights=client_weights,
                block_map=self._caga_block_map,
                loss_fn=loss_fn,
                hp=self._caga_hp,
                server_round=server_round
            )
            self._last_parameters_ndarrays = averaged
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics) if self.fit_metrics_aggregation_fn else {}
            if self._on_caga_stats is not None:
                self._on_caga_stats(server_round, caga_report)
            return (ndarrays_to_parameters(averaged), metrics_aggregated)
        if self._defense_mode != 'norm_clip':
            if self._on_defense_stats is not None and results and (self._last_parameters_ndarrays is not None):
                client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                norms = [compute_update_norm(self._last_parameters_ndarrays, arrs) for arrs in client_ndarrays]
                threshold = median_norm_clip_threshold(norms, self._clip_multiplier)
                logger.info(
                    '[Monitor] Round %d — (undefended) median_ref=%.3f | client norms=%s',
                    server_round,
                    threshold,
                    [f'{n:.3f}' for n in norms]
                )
                self._on_defense_stats(server_round, threshold, 0, norms)
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
            if parameters_aggregated is not None:
                self._last_parameters_ndarrays = parameters_to_ndarrays(parameters_aggregated)
            return (parameters_aggregated, metrics_aggregated)
        if not results:
            return (None, {})
        if self._last_parameters_ndarrays is None:
            self._last_parameters_ndarrays = parameters_to_ndarrays(results[0][1].parameters)
        client_ndarrays = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_weights = [float(fit_res.num_examples) for _, fit_res in results]
        averaged, report = clip_and_average(
            old_arrays=self._last_parameters_ndarrays,
            client_arrays=client_ndarrays,
            client_weights=client_weights,
            clip_multiplier=self._clip_multiplier,
            server_round=server_round
        )
        parameters_aggregated = ndarrays_to_parameters(averaged)
        self._last_parameters_ndarrays = averaged
        fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics) if self.fit_metrics_aggregation_fn else {}
        if self._on_defense_stats is not None:
            self._on_defense_stats(
                server_round,
                report.threshold,
                report.num_clipped,
                report.client_norms
            )
        return (parameters_aggregated, metrics_aggregated)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    seed_everything(cfg.training.seed)
    device = get_device(cfg.device)
    local_epochs = cfg.get('local_epochs', 10)
    num_rounds = cfg.get('num_rounds', 50)
    fl_cfg = cfg.get('fl', OmegaConf.create({}))
    algorithm = str(fl_cfg.get('algorithm', 'fedavg')).lower()
    fp_cfg = cfg.get('fedprox', OmegaConf.create({}))
    mu = float(fp_cfg.get('mu', 1.0))
    valid_algorithms = ('fedavg', 'fedprox')
    if algorithm not in valid_algorithms:
        raise ValueError(f"Unknown fl.algorithm='{algorithm}'. Choose from: {', '.join(valid_algorithms)}")
    logger.info('FL algorithm: %s%s', algorithm, f' (μ={mu})' if algorithm == 'fedprox' else '')
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f'fl_{algorithm}_seed{cfg.training.seed}',
        config=cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        tags=['federated', 'flower', 'feature-bootstrapping', 'shared-global-graph', algorithm],
        group='federated_caga',
        mode=cfg.wandb.mode
    )
    data, evaluator = load_dataset(root=cfg.dataset.root)
    assert data.edge_index is not None
    assert data.num_nodes is not None
    split = get_edge_split(root=cfg.dataset.root)
    num_nodes: int = data.num_nodes
    global_edge_index: torch.Tensor = data.edge_index
    feat_cfg = cfg.get('fixed_features', {})
    num_anchors_feat = int(feat_cfg.get('num_anchors', 32))
    num_spectral_feat = int(feat_cfg.get('num_spectral', 16))
    feat_cache_path = os.path.join(
        cfg.dataset.root,
        f'fixed_features_cache_n{num_nodes}_a{num_anchors_feat}_s{num_spectral_feat}_seed{cfg.training.seed}.pt'
    )
    if os.path.exists(feat_cache_path):
        logger.info('Loading cached fixed global features from %s ...', feat_cache_path)
        node_features = torch.load(feat_cache_path, weights_only=False)
    else:
        logger.info(
            'Computing fixed global features (anchors=%d, spectral=%d) ...',
            num_anchors_feat,
            num_spectral_feat
        )
        node_features = compute_fixed_global_features(
            global_edge_index.cpu(),
            num_nodes,
            num_anchors=num_anchors_feat,
            num_spectral=num_spectral_feat,
            seed=cfg.training.seed
        )
        os.makedirs(os.path.dirname(feat_cache_path) or '.', exist_ok=True)
        torch.save(node_features, feat_cache_path)
        logger.info('Cached fixed global features to %s for future runs.', feat_cache_path)
    feat_dim = node_features.shape[1]
    logger.info(
        'Fixed feature dimension: %d → projected to hidden_dim=%d',
        feat_dim,
        cfg.model.hidden_dim
    )
    dist_enc = None
    if cfg.dataset.get('num_anchors', 0) > 0:
        dist_enc = compute_distance_encoding(
            global_edge_index.cpu(),
            num_nodes,
            num_anchors=cfg.dataset.num_anchors,
            seed=cfg.training.seed
        )
    num_clients = cfg.get('num_clients', 5)
    graph = global_edge_index
    atc_mapping = None
    atc_path = os.path.join(cfg.dataset.root, 'ogbl_ddi', 'atc_mapping.pt')
    if os.path.exists(atc_path):
        atc_mapping = torch.load(atc_path)
        logger.info('Loaded ATC mapping from %s for observation sampling.', atc_path)
    else:
        logger.warning(
            'No ATC mapping found at %s (run scripts/prepare_atc_mapping.py to generate one) — falling back to class-agnostic observation sampling. Heterogeneity will still differ per client, just not along therapeutic-area lines.',
            atc_path
        )
    obs_cfg = cfg.get('local_observations', {})
    # Root edges are excluded from client observations, but remain in the public graph.
    _early_defense_mode = cfg.get('defense', {}).get('mode', 'none')
    root_edges: Optional[torch.Tensor] = None
    if _early_defense_mode in ('caga', 'fltrust', 'root_client'):
        _defense_cfg_early = cfg.get('defense', {})
        if 'root_frac' in _defense_cfg_early:
            _root_frac = float(_defense_cfg_early['root_frac'])
        else:
            _root_frac = 0.12
        train_edges_full = split['train']['edge']
        _root_gen = torch.Generator().manual_seed(int(cfg.training.seed) + 999)
        _perm = torch.randperm(train_edges_full.size(0), generator=_root_gen)
        _n_root = max(50, int(_root_frac * train_edges_full.size(0)))
        root_edges = train_edges_full[_perm[:_n_root]].clone()
        train_edges_for_clients = train_edges_full[_perm[_n_root:]].clone()
        logger.info(
            "Root dataset reserved (defense_mode=%s): %d edges (%.2f%% of %d total train edges) held out from the client sampling pool — these edges are never included in any client's local_observations sample below.",
            _early_defense_mode,
            _n_root,
            100.0 * _n_root / train_edges_full.size(0),
            train_edges_full.size(0)
        )
    else:
        train_edges_for_clients = split['train']['edge']
    client_datasets = generate_client_observations(
        train_edges=train_edges_for_clients,
        num_clients=num_clients,
        atc_mapping=atc_mapping,
        observations_per_client=obs_cfg.get('observations_per_client', None),
        dirichlet_alpha=obs_cfg.get('dirichlet_alpha', 0.5),
        seed=obs_cfg.get('seed', None) or cfg.training.seed
    )
    client_trainers: List[Trainer] = [build_client_trainer(cid=cid, graph=graph, node_features=node_features, train_edges=client_datasets[cid].edges, valid_split=split['valid'], test_split=split['test'], num_nodes=num_nodes, evaluator=evaluator, cfg=cfg, device=device, node_dist_enc=dist_enc) for cid in range(num_clients)]
    root_trainer: Optional[Trainer] = None
    _root_device = torch.device('cpu')
    _want_root_dev = str(cfg.get('defense', {}).get('root_device', 'cpu'))
    if _want_root_dev != 'cpu':
        if bool(cfg.get('perf', {}).get('dense_adj', False)) and torch.cuda.is_available():
            _root_device = torch.device(_want_root_dev)
        else:
            logger.warning(
                'defense.root_device=%s ignored (needs perf.dense_adj=true and CUDA); using CPU.',
                _want_root_dev
            )
    if root_edges is not None:
        root_trainer = build_client_trainer(
            cid=-1,
            graph=graph,
            node_features=node_features,
            train_edges=root_edges,
            valid_split=split['valid'],
            test_split=split['test'],
            num_nodes=num_nodes,
            evaluator=evaluator,
            cfg=cfg,
            device=_root_device,
            node_dist_enc=dist_enc
        )
    caga_block_map: Optional[Dict[str, List[int]]] = None
    if _early_defense_mode in ('caga',):
        assert root_trainer is not None
        from src.training.caga_probe import compute_block_index_map
        caga_block_map = compute_block_index_map(root_trainer.encoder, root_trainer.predictor)
    poison_cfg = cfg.get('poisoning', {})
    poison_event_specs = poison_cfg.get('events', []) or []
    poison_events: List[PoisonEvent] = [PoisonEvent(client_id=int(spec['client_id']), start_round=int(spec['start_round']), suppress={int(k): float(v) for k, v in spec.get('suppress', {}).items()}, emerge={int(k): float(v) for k, v in spec.get('emerge', {}).items()}, kind=spec.get('kind', 'sudden'), transition_rounds=int(spec.get('transition_rounds', 10)), update_scale=float(spec.get('update_scale', 1.0))) for spec in poison_event_specs]
    fraction_malicious = float(poison_cfg.get('fraction_malicious', 0.0) or 0.0)
    if fraction_malicious > 0.0:
        already_malicious = {ev.client_id for ev in poison_events}
        poison_events += [ev for ev in build_poison_events_from_fraction(num_clients=num_clients, fraction_malicious=fraction_malicious, start_round=int(poison_cfg.get('start_round', 1)), suppress={int(k): float(v) for k, v in poison_cfg.get('suppress', {}).items()}, emerge={int(k): float(v) for k, v in poison_cfg.get('emerge', {}).items()}, kind=poison_cfg.get('kind', 'sudden'), transition_rounds=int(poison_cfg.get('transition_rounds', 10)), update_scale=float(poison_cfg.get('update_scale', 1.0)), seed=poison_cfg.get('seed', None) or cfg.training.seed) if ev.client_id not in already_malicious]
    poisoning_simulator: PoisoningSimulator | None = None
    if poison_events:
        poisoning_simulator = PoisoningSimulator(
            initial_datasets=client_datasets,
            train_edges=split['train']['edge'],
            atc_mapping=atc_mapping,
            events=poison_events,
            observations_per_client=obs_cfg.get('observations_per_client', None),
            seed=poison_cfg.get('seed', None) or cfg.training.seed
        )
        logger.info('Data poisoning enabled — %d malicious client(s) configured.', len(poison_events))
    else:
        logger.info('No poisoning events configured — running honest %s baseline.', algorithm)
    data_drift_cfg = cfg.get('data_drift', {})
    data_drift_event_specs = data_drift_cfg.get('events', []) or []
    data_drift_events: List[FeatureDriftEvent] = [FeatureDriftEvent(client_id=int(spec['client_id']), start_round=int(spec['start_round']), feature_dims=[int(d) for d in spec['feature_dims']], shift_magnitude=float(spec.get('shift_magnitude', 2.0)), noise_std=float(spec.get('noise_std', 0.0)), target_node_frac=float(spec.get('target_node_frac', 0.3)), node_selection=spec.get('node_selection', 'high_degree'), kind=spec.get('kind', 'sudden'), transition_rounds=int(spec.get('transition_rounds', 10))) for spec in data_drift_event_specs]
    data_drift_simulator: DataDriftSimulator | None = None
    if data_drift_events:
        data_drift_simulator = DataDriftSimulator(
            node_features=node_features,
            num_clients=num_clients,
            events=data_drift_events,
            seed=data_drift_cfg.get('seed', None) or cfg.training.seed
        )
        logger.info('Data drift enabled — %d drifting client(s) configured.', len(data_drift_events))
    else:
        logger.info(
            'No data-drift events configured — running stable %s baseline (all client feature distributions fixed for the entire run).',
            algorithm
        )
    defense_cfg = cfg.get('defense', {})
    defense_mode = defense_cfg.get('mode', 'none')
    clip_multiplier = float(defense_cfg.get('clip_multiplier', 1.5))
    if 'root_frac' in defense_cfg:
        root_frac_used = float(defense_cfg['root_frac'])
    else:
        root_frac_used = 0.12
    caga_hp = None
    if defense_mode == 'caga':
        from src.training.caga import CAGAHyperparams
        caga_hp = CAGAHyperparams(
            tol_z=float(defense_cfg.get('caga_tol_z', 2.0)),
            tol_abs=float(defense_cfg.get('caga_tol_abs', 0.0001)),
            tol_rel=float(defense_cfg.get('caga_tol_rel', 0.0)),
            probe_size=int(defense_cfg.get('caga_probe_size', 4000)),
            gate_blocks=bool(defense_cfg.get('caga_gate_blocks', True)),
            final_check=bool(defense_cfg.get('caga_final_check', True)),
            max_sweeps=int(defense_cfg.get('caga_max_sweeps', 2)),
            max_rel_increase=float(defense_cfg.get('caga_max_rel_increase', 0.1)),
            warmup_rounds=int(defense_cfg.get('caga_warmup_rounds', 0))
        )
        logger.info(
            'Poisoning+drift recovery ENABLED — CAGA defense (src/training/caga.py, docs/METHOD.md): %s, root_frac=%.3f',
            caga_hp,
            root_frac_used
        )
    if defense_mode not in ('none', 'norm_clip', 'caga', 'fltrust', 'root_client'):
        raise ValueError(f"Unknown defense.mode {defense_mode!r} — expected 'none', 'norm_clip', 'caga', 'fltrust' or 'root_client'.")
    server_update: Optional[Callable[[int, List[np.ndarray]], List[np.ndarray]]] = None
    if defense_mode in ('fltrust', 'root_client'):
        assert root_trainer is not None
        # The server's reference update uses the same local training procedure as the clients.
        _root_client = FedDDIClient(
            cid='-1',
            trainer=root_trainer,
            local_epochs=local_epochs,
            device=_root_device,
            algorithm=algorithm,
            mu=mu,
            num_rounds=num_rounds
        )

        # The server trains on a fresh client-sized sample of root edges each round and takes as many steps
        # as a client: FLTrust rescales client updates to this update's norm, and root_client weights it as one.
        _fltrust_obs = defense_cfg.get('fltrust_root_obs', None) or obs_cfg.get('observations_per_client', None)
        _fltrust_root_edges = root_trainer.split['train']['edge']

        def server_update(rnd: int, arrays: List[np.ndarray]) -> List[np.ndarray]:
            if _fltrust_obs:
                g = torch.Generator().manual_seed(int(cfg.training.seed) * 100003 + rnd)
                pick = torch.randint(0, _fltrust_root_edges.size(0), (int(_fltrust_obs),), generator=g)
                root_trainer.split['train']['edge'] = _fltrust_root_edges[pick]
            new_arrays, _, _ = _root_client.fit(arrays, {'server_round': rnd})
            return new_arrays
        logger.info(
            'Server root update ENABLED — %s (root_frac=%.3f), server trains on %s',
            'FLTrust defense (src/training/fltrust.py)' if defense_mode == 'fltrust' else 'root_client baseline',
            root_frac_used,
            _root_device
        )
    if defense_mode == 'norm_clip':
        logger.info(
            'Poisoning recovery ENABLED — norm_clip defense, clip_multiplier=%.2f',
            clip_multiplier
        )
    elif defense_mode == 'none':
        logger.info('No defense configured — running undefended %s.', algorithm)
    run_tag = algorithm
    if algorithm == 'fedprox':
        run_tag += f'_mu{mu}'
    _run_has_corruption = bool(poison_events) or bool(data_drift_events)
    if poison_events and data_drift_events:
        run_tag += '_poison+datadrift'
    elif poison_events:
        run_tag += '_poison'
    elif data_drift_events:
        run_tag += '_datadrift'
    else:
        run_tag += '_baseline'
    if defense_mode == 'norm_clip':
        run_tag += '_defended'
    elif defense_mode == 'caga':
        run_tag += '_defended_caga'
    elif defense_mode == 'fltrust':
        run_tag += '_defended_fltrust'
    elif defense_mode == 'root_client':
        run_tag += '_root_client'
    csv_dir = cfg.get('csv_output_dir', '.')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'results_{run_tag}_seed{cfg.training.seed}.csv')
    logger.info('Per-round CSV will be written to %s', csv_path)
    metrics_history: Dict[int, Dict[str, Any]] = {}
    csv_fieldnames = [
        'round',
        'train_loss',
        'val_hits20',
        'test_hits20',
        'federated_val_hits20',
        'federated_local_val_hits20',
        'poison_attack_mean_prob',
        'poison_attack_n',
        'poison_fabricated_mean_prob',
        'poison_suppressed_mean_prob',
        'drift_region_hits20',
        'drift_region_n',
        'defense_clip_threshold',
        'defense_num_clipped',
        'defense_client_norms',
        'non_drift_region_hits20',
        'non_drift_region_n',
        'caga_client_keys',
        'caga_alpha',
        'caga_scale_phi',
        'caga_scale_convs',
        'caga_scale_mlp',
        'caga_loss_delta',
        'caga_verdict',
        'caga_loss_before',
        'caga_loss_after',
        'caga_final_gamma',
        'fltrust_client_keys',
        'fltrust_cosine',
        'fltrust_trust',
        'fltrust_client_norms',
        'fltrust_server_norm'
    ]

    def _flush_csv() -> None:
        rows = [metrics_history[r] for r in sorted(metrics_history)]
        if not rows:
            return
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in csv_fieldnames})

    def _record_server_eval(rnd: int, val_h20: float, test_h20: float) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row['val_hits20'] = val_h20
        row['test_hits20'] = test_h20
        _flush_csv()

    def _record_fit(rnd: int, aggregated: Dict[str, float]) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row['train_loss'] = aggregated.get('loss', '')
        _flush_csv()

    def _record_defense_stats(
        rnd: int,
        threshold: float,
        num_clipped: int,
        client_norms: List[float]
    ) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row['defense_clip_threshold'] = threshold
        row['defense_num_clipped'] = num_clipped
        import json as _json
        row['defense_client_norms'] = _json.dumps([round(n, 4) for n in client_norms])
        _flush_csv()

    def _record_caga_stats(rnd: int, report: Any) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row.update(report.as_row_fields())
        _flush_csv()

    def _record_fltrust_stats(rnd: int, report: Any) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row.update(report.as_row_fields())
        _flush_csv()

    def _record_federated_eval(rnd: int, aggregated: Dict[str, float]) -> None:
        row = metrics_history.setdefault(rnd, {'round': rnd})
        row['federated_val_hits20'] = aggregated.get('Hits@20', '')
        row['federated_local_val_hits20'] = aggregated.get('local_Hits@20', '')
        if 'poison_attack_mean_prob' in aggregated:
            row['poison_attack_mean_prob'] = aggregated['poison_attack_mean_prob']
            row['poison_attack_n'] = aggregated.get('poison_attack_n', '')
        for key in ('poison_fabricated_mean_prob', 'poison_suppressed_mean_prob'):
            if key in aggregated:
                row[key] = aggregated[key]
        if 'drift_region_Hits@20' in aggregated:
            row['drift_region_hits20'] = aggregated['drift_region_Hits@20']
            row['drift_region_n'] = aggregated.get('drift_region_n', '')
        if 'non_drift_region_Hits@20' in aggregated:
            row['non_drift_region_hits20'] = aggregated['non_drift_region_Hits@20']
            row['non_drift_region_n'] = aggregated.get('non_drift_region_n', '')
        _flush_csv()
    _trainer_path = None
    if bool(cfg.get('perf', {}).get('actor_cache', False)):
        import atexit
        import shutil
        import tempfile
        import cloudpickle
        _tdir = tempfile.mkdtemp(prefix='feddi_trainers_')
        atexit.register(shutil.rmtree, _tdir, ignore_errors=True)
        _trainer_path = os.path.join(_tdir, 'client_trainers.pkl')
        with open(_trainer_path, 'wb') as _f:
            cloudpickle.dump(client_trainers, _f)
        logger.info(
            '[perf] actor_cache ON: %d client trainers saved once to %s (%.1f MB); each Ray actor loads them once.',
            len(client_trainers),
            _trainer_path,
            os.path.getsize(_trainer_path) / 1000000.0
        )
    if _trainer_path is not None:

        def _get_trainer(i: int, _p: str=_trainer_path):
            from src.training.actor_cache import get_trainers
            return get_trainers(_p)[i]
    else:

        def _get_trainer(i: int, _t: list=client_trainers):
            return _t[i]

    def client_fn(context: Context) -> Client:
        c_id_int = int(context.node_config['partition-id'])
        return FedDDIClient(
            cid=str(c_id_int),
            trainer=_get_trainer(c_id_int),
            local_epochs=local_epochs,
            device=device,
            algorithm=algorithm,
            mu=mu,
            num_rounds=num_rounds,
            c_id_int=c_id_int,
            poisoning_simulator=poisoning_simulator,
            data_drift_simulator=data_drift_simulator
        ).to_client()
    client_app = ClientApp(client_fn=client_fn)
    try:
        import cloudpickle as _cp
        import time as _time
        _t = _time.perf_counter()
        _n = len(_cp.dumps(client_app))
        logger.info(
            '[perf] ClientApp (closure incl. all client trainers) pickles to %.1f MB in %.2fs - this is re-sent on every client call',
            _n / 1000000.0,
            _time.perf_counter() - _t
        )
    except Exception as _e:
        logger.info('[perf] ClientApp pickle-size probe skipped: %r', _e)
    global_encoder = GraphSAGEEncoder(
        num_nodes=num_nodes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        normalize=cfg.model.normalize,
        node_features=node_features
    ).to(device)
    global_predictor = NeuralLinkPredictor(
        emb_dim=cfg.model.hidden_dim,
        hidden_dim=cfg.predictor.hidden_dim,
        num_layers=cfg.predictor.num_layers,
        dropout=cfg.predictor.dropout,
        dist_dim=cfg.dataset.get('num_anchors', 0)
    ).to(device)
    full_edge_index = global_edge_index.to(device)
    agg_params: List[np.ndarray] = []
    for name, param in global_encoder.named_parameters():
        if _is_aggregated_encoder_param(name):
            agg_params.append(param.cpu().detach().numpy())
    for name, param in global_predictor.named_parameters():
        if _is_aggregated_predictor_param(name):
            agg_params.append(param.cpu().detach().numpy())
    initial_parameters = ndarrays_to_parameters(agg_params)
    server_eval_fn = build_server_evaluate_fn(
        encoder=global_encoder,
        predictor=global_predictor,
        full_edge_index=full_edge_index,
        split=split,
        evaluator=evaluator,
        device=device,
        batch_size=cfg.training.batch_size,
        node_dist_enc=dist_enc.to(device) if dist_enc is not None else None,
        using_shared_node_features=True,
        on_round_eval=_record_server_eval
    )
    best_tracker: Dict[str, float] = {}

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = CAGAStrategy(
            algorithm=algorithm,
            num_total_clients=num_clients,
            defense_mode=defense_mode,
            clip_multiplier=clip_multiplier,
            on_defense_stats=_record_defense_stats,
            root_trainer=root_trainer,
            caga_block_map=caga_block_map,
            caga_hp=caga_hp,
            on_caga_stats=_record_caga_stats,
            server_update=server_update,
            on_fltrust_stats=_record_fltrust_stats,
            fraction_fit=1.0,
            fraction_evaluate=cfg.get('fraction_evaluate', 1.0),
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn,
            fit_metrics_aggregation_fn=make_metrics_aggregation_fn('fit', on_round=_record_fit),
            evaluate_metrics_aggregation_fn=make_metrics_aggregation_fn(
                'evaluate',
                best_tracker=best_tracker,
                on_round=_record_federated_eval
            )
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=num_rounds))
    server_app = ServerApp(server_fn=server_fn)
    logger.info(
        'Starting Flower simulation — %s | Feature Bootstrapping | shared global graph | %d clients | %d rounds | %d local epochs/round | feat_dim=%d',
        algorithm.upper(),
        num_clients,
        num_rounds,
        local_epochs,
        feat_dim
    )
    ray_gpus = 1 if device.type == 'cuda' and torch.cuda.is_available() else 0
    # Full-graph client models share one GPU; avoid overlapping their peak allocations.
    MAX_SAFE_CONCURRENT_CLIENTS = 1
    available_cpus = os.cpu_count() or 1
    num_cpus_per_client = max(1, available_cpus // MAX_SAFE_CONCURRENT_CLIENTS)
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config={'client_resources': {'num_cpus': num_cpus_per_client, 'num_gpus': 1.0 / MAX_SAFE_CONCURRENT_CLIENTS if ray_gpus > 0 else 0.0}}
    )
    logger.info('Flower simulation finished — %d rounds completed.', num_rounds)
    logger.info('Per-round CSV written to %s (%d rows).', csv_path, len(metrics_history))
    plotting_cfg = cfg.get('plotting', None)
    if plotting_cfg and plotting_cfg.get('enabled', True):
        val_hits20 = []
        test_hits20 = []
        dist_val_hits20 = []
        dist_local_history = []
        for rnd in sorted(metrics_history.keys()):
            row = metrics_history[rnd]
            if 'val_hits20' in row and row['val_hits20'] is not None and (row['val_hits20'] != ''):
                val_hits20.append((rnd, float(row['val_hits20'])))
            if 'test_hits20' in row and row['test_hits20'] is not None and (row['test_hits20'] != ''):
                test_hits20.append((rnd, float(row['test_hits20'])))
            if 'federated_val_hits20' in row and row['federated_val_hits20'] != '':
                dist_val_hits20.append((rnd, float(row['federated_val_hits20'])))
            if 'federated_local_val_hits20' in row and row['federated_local_val_hits20'] != '':
                dist_local_history.append((rnd, float(row['federated_local_val_hits20'])))
        plot_history = {
            'val_hits20': val_hits20,
            'test_hits20': test_hits20,
            'dist_val_hits20': dist_val_hits20,
            'dist_local_hits20': dist_local_history
        }
        from src.utils.plotting import plot_federated_learning
        plot_federated_learning(
            history_dict=plot_history,
            save_dir=plotting_cfg.get('save_dir', './plots'),
            show=plotting_cfg.get('show', True)
        )
    if 'val_hits20' in best_tracker:
        logger.info(
            'Best federated Val Hits@20 = %.4f at round %d.',
            best_tracker['val_hits20'],
            int(best_tracker['round'])
        )
        wandb.summary['best_federated_val_hits20'] = best_tracker['val_hits20']
        wandb.summary['best_federated_val_round'] = int(best_tracker['round'])
    else:
        logger.warning('No federated evaluation metrics recorded — check fraction_evaluate > 0.')
    wandb.finish()
if __name__ == '__main__':
    main()
