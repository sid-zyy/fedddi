"""
main_fl.py  (Strategy 1 — Feature Bootstrapping)
──────────────────────────────────────────────────
Federated training entry point for FedDDI-GNN.

What changed from the broken original
──────────────────────────────────────
The original code had one fundamental problem and several secondary ones:

PRIMARY BUG — Orthogonal Embedding Spaces
  Each client kept a PRIVATE nn.Embedding table as its node-representation
  source. FedAvg aggregated the SAGEConv layers and the MLP decoder, but
  those layers received *different*, *unaligned* input vectors on every
  client. Averaging gradients of a function f(embed_A(v)) and f(embed_B(v))
  where embed_A and embed_B are independent random initialisations in Rᴴ
  is mathematically equivalent to averaging random directional updates — the
  result converges to nothing meaningful. This alone explains Hits@20 ≈ 0.

SECONDARY — server_diag_eval_fn was wrong in the default mode
  The server built a global encoder with fixed topo features fed through a
  feature_proj that was NEVER updated (clients in embedding mode don't
  contribute a feature_proj). Its node representations were Xavier-random
  noise for the entire run, producing random logit gaps, hence ~0 Hits@20.
  This path is preserved as a diagnostic but now correctly gated behind
  `using_shared_node_features`.

Strategy 1 Fix — Feature Bootstrapping
  compute_fixed_global_features() produces a fixed [N, D] tensor that is
  identical on every client and the server. This replaces the private
  embedding table. The feature_proj Linear is now part of the aggregation
  parameter set (it was not before). Because ALL parties share the same
  input feature space, FedAvg averaging of (feature_proj + convs + MLP)
  gradients is now meaningful: the aggregated model maps a globally-shared
  feature → embedding → score, and evaluating it with any client's encoder
  gives scores in the same logit space. This restores the mathematical
  soundness of the federated evaluation.

Param aggregation contract (matches fl_client.py constants exactly)
  Aggregated :  encoder.feature_proj.*  +  encoder.convs.*  +  predictor.mlp.*
  NOT aggregated: (nothing — there is no private embedding in Strategy 1)

Graph ownership model (no partitioning)
  There is no client-specific graph construction in this project. Every
  client and the server hold a reference to the SAME global DDI graph
  (`global_edge_index`) — the public, already-documented ogbl-ddi /
  DrugBank-scale interaction catalog. Nothing about the graph's topology
  is hidden, subset, or owned differently by any client: a published
  "drug A interacts with drug B" record is public pharmacological
  knowledge, not a patient secret, so there is no privacy boundary to
  enforce by partitioning it. The privacy boundary in this project lives
  on each client's PRIVATE local clinical observations (which training
  interactions it samples and trains on), not on the graph — see
  `src/data/local_observations.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Tuple, cast

import hydra
import numpy as np
import torch
import wandb
from flwr.client import Client, ClientApp  # type: ignore
from flwr.common import Context, Scalar, ndarrays_to_parameters  # type: ignore
from flwr.server import ServerApp, ServerAppComponents, ServerConfig  # type: ignore
from flwr.server.strategy import FedAvg  # type: ignore
from flwr.simulation import run_simulation  # type: ignore
from omegaconf import DictConfig, OmegaConf
from torch_geometric.utils import to_undirected  # type: ignore

from src.data.dataset import get_edge_split, load_dataset
from src.data.drift import ConceptDriftSimulator, DriftEvent
from src.data.local_observations import generate_client_observations
from src.data.features import (
    compute_distance_encoding,
    compute_fixed_global_features,
    compute_topo_features,
)
from src.models.encoder import GraphSAGEEncoder
from src.models.predictor import NeuralLinkPredictor
from src.training.evaluate import evaluate
from src.training.fl_client import (
    FedDDIClient,
    _is_aggregated_encoder_param,
    _is_aggregated_predictor_param,
)
from src.training.trainer import Trainer, build_client_trainer
from src.utils.utils import get_device, seed_everything, setup_logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Server-side evaluation
# ─────────────────────────────────────────────────────────────────────────────


def build_server_evaluate_fn(
    encoder: GraphSAGEEncoder,
    predictor: NeuralLinkPredictor,
    full_edge_index: torch.Tensor,
    split: Dict[str, Any],
    evaluator: Any,
    device: torch.device,
    batch_size: int,
    node_dist_enc: torch.Tensor | None = None,
    using_shared_node_features: bool = False,
) -> Callable[
    [int, List[np.ndarray], Dict[str, Scalar]],
    Tuple[float, Dict[str, Scalar]] | None,
]:
    """Build the server-side evaluation function.

    When using_shared_node_features=True (Strategy 1):
      The server encoder receives the same fixed global features as every
      client. Its feature_proj IS part of the aggregated parameter set and is
      updated every round. This function therefore produces the REAL global
      validation Hits@20 and is logged as the primary server metric.

    When using_shared_node_features=False (legacy / ablation):
      The server's feature_proj is fed topo features but is never updated by
      aggregation — results are diagnostic noise only, labelled accordingly.

    This function restores aggregated parameters into the server's encoder and
    predictor using the SAME key-filter as fl_client.py's get/set_parameters,
    guaranteeing that the parameter ordering is identical.
    """

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, Dict[str, Scalar]] | None:

        # ── Restore aggregated parameters (same filter as fl_client.py) ─────
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

        # ── Score ────────────────────────────────────────────────────────────
        val_hits = evaluate(
            encoder, predictor, full_edge_index,
            split["valid"]["edge"], split["valid"]["edge_neg"],
            evaluator, batch_size, device, node_dist_enc,
        )
        test_hits = evaluate(
            encoder, predictor, full_edge_index,
            split["test"]["edge"], split["test"]["edge_neg"],
            evaluator, batch_size, device, node_dist_enc,
        )

        h20_val  = float(val_hits.get("Hits@20", 0.0))
        h20_test = float(test_hits.get("Hits@20", 0.0))

        if using_shared_node_features:
            tag = "server_eval"
            label = ""
        else:
            tag = "server_diag"
            label = " [diagnostic-only — feature_proj not aggregated in this mode]"

        logger.info(
            "Round %d — %s Val Hits@20: %.4f | Test Hits@20: %.4f%s",
            server_round, tag, h20_val, h20_test, label,
        )
        wandb.log({
            "round": server_round,
            f"{tag}_val_hits20":  h20_val,
            f"{tag}_test_hits20": h20_test,
        })
        return 0.0, {f"{tag}_Hits@20": h20_val}

    return evaluate_fn


def make_metrics_aggregation_fn(
    kind: str,
    best_tracker: Dict[str, float] | None = None,
) -> Callable[[List[Tuple[int, Dict[str, Scalar]]]], Dict[str, Scalar]]:
    """Weighted-average aggregation for fit or evaluate metrics.

    Logs every round to both the Python logger and W&B. For kind="evaluate",
    also updates best_tracker in-place when a new best is seen.
    """
    round_counter = {"n": 0}

    def aggregation_fn(
        results: List[Tuple[int, Dict[str, Scalar]]],
    ) -> Dict[str, Scalar]:
        if not results:
            return {}

        total_examples = sum(n for n, _ in results)
        if total_examples == 0:
            return {}

        aggregated: Dict[str, float] = {}
        for num_examples, metrics in results:
            for key, value in metrics.items():
                if key == "cid":
                    continue
                if isinstance(value, (int, float)):
                    aggregated[key] = aggregated.get(key, 0.0) + float(value) * num_examples

        for key in aggregated:
            aggregated[key] /= total_examples

        round_counter["n"] += 1
        rnd = round_counter["n"]

        if kind == "evaluate":
            val_h20 = aggregated.get("Hits@20", 0.0)
            logger.info(
                "Round %d — federated Val Hits@20: %.4f",
                rnd, val_h20,
            )
            wandb.log({
                "round": rnd,
                "federated_val_hits20": val_h20,
            })
            if best_tracker is not None and val_h20 > best_tracker.get("val_hits20", -1.0):
                best_tracker["val_hits20"] = val_h20
                best_tracker["round"]      = float(rnd)

        elif kind == "fit":
            loss = aggregated.get("loss", float("nan"))
            logger.info("Round %d — federated train loss: %.4f", rnd, loss)
            wandb.log({"round": rnd, "federated_train_loss": loss})

        return cast(Dict[str, Scalar], aggregated)

    return aggregation_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.log_level)
    seed_everything(cfg.training.seed)
    device = get_device(cfg.device)

    local_epochs = cfg.get("local_epochs", 10)
    num_rounds   = cfg.get("num_rounds", 50)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"fl_strategy1_seed{cfg.training.seed}",
        config=cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        tags=["federated", "flower", "strategy1-feature-bootstrapping",
              "shared-global-graph"],
        group="federated_strategy1",
        mode=cfg.wandb.mode,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    data, evaluator = load_dataset(root=cfg.dataset.root)
    assert data.edge_index is not None
    assert data.num_nodes is not None
    split = get_edge_split(root=cfg.dataset.root)

    num_nodes: int         = data.num_nodes
    global_edge_index: torch.Tensor = data.edge_index

    # ── Strategy 1: compute fixed global features (shared across all clients) ─
    # These are topology-derived from the PUBLIC training edge_index.
    # Every client and the server compute them identically.
    feat_cfg = cfg.dataset.get("fixed_features", {})
    num_anchors_feat  = int(feat_cfg.get("num_anchors",  64))
    num_spectral_feat = int(feat_cfg.get("num_spectral", 32))

    logger.info(
        "Strategy 1: computing fixed global features "
        "(anchors=%d, spectral=%d) ...", num_anchors_feat, num_spectral_feat
    )
    node_features = compute_fixed_global_features(
        global_edge_index.cpu(),
        num_nodes,
        num_anchors=num_anchors_feat,
        num_spectral=num_spectral_feat,
        seed=cfg.training.seed,
    )  # [N, 2 + num_anchors_feat + num_spectral_feat]

    feat_dim = node_features.shape[1]
    logger.info("Fixed feature dimension: %d → projected to hidden_dim=%d",
                feat_dim, cfg.model.hidden_dim)

    # ── Optional distance encoding for MLP decoder ────────────────────────────
    dist_enc = None
    if cfg.dataset.get("num_anchors", 0) > 0:
        dist_enc = compute_distance_encoding(
            global_edge_index.cpu(), num_nodes,
            num_anchors=cfg.dataset.num_anchors, seed=cfg.training.seed,
        )

    # ── Shared global graph ────────────────────────────────────────────────────
    # Every client trains on the SAME complete, public DDI graph — there is no
    # per-client graph construction. `graph` below is one shared reference,
    # not a copy: the message-passing topology is identical for every client.
    num_clients = cfg.get("num_clients", 5)
    graph = global_edge_index

    # ── Heterogeneous local observations (NOT graph partitioning) ──────────────
    # Every client still trains GraphSAGE message-passing over the complete
    # shared `graph` above. What differs per client is which TRAINING
    # OBSERVATIONS (edges used as positive training targets) it sees — drawn
    # via ATC-aware Dirichlet sampling, matching the FedDrift proposal's
    # "Hospital A: 70% Cardiovascular / 20% Diabetes / 10% Oncology" framing.
    # See src/data/local_observations.py for the full design rationale.
    atc_mapping = None
    atc_path = os.path.join(cfg.dataset.root, "ogbl_ddi", "atc_mapping.pt")
    if os.path.exists(atc_path):
        atc_mapping = torch.load(atc_path)
        logger.info("Loaded ATC mapping from %s for observation sampling.", atc_path)
    else:
        logger.warning(
            "No ATC mapping found at %s (run scripts/prepare_atc_mapping.py to "
            "generate one) — falling back to class-agnostic observation "
            "sampling. Heterogeneity will still differ per client, just not "
            "along therapeutic-area lines.",
            atc_path,
        )

    obs_cfg = cfg.get("local_observations", {})
    client_datasets = generate_client_observations(
        train_edges=split["train"]["edge"],
        num_clients=num_clients,
        atc_mapping=atc_mapping,
        observations_per_client=obs_cfg.get("observations_per_client", None),
        dirichlet_alpha=obs_cfg.get("dirichlet_alpha", 0.5),
        seed=obs_cfg.get("seed", None) or cfg.training.seed,
    )

    # ── Build per-client trainers ──────────────────────────────────────────────
    # build_client_trainer() is the single place defining the federated-client
    # contract (shared graph/features, own model, own optimizer) — see its
    # docstring. Each client's own local observations now come from
    # client_datasets[cid] above instead of the shared global train_edges.
    client_trainers: List[Trainer] = [
        build_client_trainer(
            cid=cid,
            graph=graph,
            node_features=node_features,
            train_edges=client_datasets[cid].edges,
            valid_split=split["valid"],
            test_split=split["test"],
            num_nodes=num_nodes,
            evaluator=evaluator,
            cfg=cfg,
            device=device,
            node_dist_enc=dist_enc,
        )
        for cid in range(num_clients)
    ]

    # ── Concept drift simulator (Step 5) ──────────────────────────────────────
    # Activated only when the config contains a non-empty `drift.events` list.
    # When disabled, all client distributions remain stable for the entire run
    # — a clean FedAvg baseline. The graph is NEVER modified by the simulator;
    # only each client's local training observations (train split) evolve.
    drift_cfg = cfg.get("drift", {})
    drift_event_specs = drift_cfg.get("events", []) or []

    drift_events: List[DriftEvent] = []
    for spec in drift_event_specs:
        # Each spec is a dict/DictConfig:
        #   {client_id, start_round, target_mixture,
        #    kind (optional, default "sudden"),
        #    transition_rounds (optional, default 10)}
        drift_events.append(
            DriftEvent(
                client_id=int(spec["client_id"]),
                start_round=int(spec["start_round"]),
                target_mixture={
                    int(k): float(v)
                    for k, v in spec["target_mixture"].items()
                },
                kind=spec.get("kind", "sudden"),
                transition_rounds=int(spec.get("transition_rounds", 10)),
            )
        )

    drift_simulator: ConceptDriftSimulator | None = None
    if drift_events:
        drift_simulator = ConceptDriftSimulator(
            initial_datasets=client_datasets,
            train_edges=split["train"]["edge"],
            atc_mapping=atc_mapping,
            events=drift_events,
            observations_per_client=obs_cfg.get("observations_per_client", None),
            seed=drift_cfg.get("seed", None) or cfg.training.seed,
        )
        logger.info(
            "Concept drift enabled — %d event(s) configured.",
            len(drift_events),
        )
    else:
        logger.info(
            "No drift events configured — running stable FedAvg baseline "
            "(all client distributions fixed for the entire run).",
        )

    # ── Drift-aware FedAvg subclass ───────────────────────────────────────────
    # The round_tracker-in-a-closure approach does NOT work with Flower's Ray
    # simulation backend: client_fn is cloudpickled once when the Ray actor is
    # created, freezing round_tracker at {"current_round": 0}. Mutations in the
    # main process are invisible to the actor. The fix is to inject the round
    # number through Flower's OWN per-round config mechanism: FedAvg.configure_fit
    # already sends a config dict to every client, so we subclass to add
    # "server_round" to that dict. The dict is re-serialised fresh every round
    # (it's part of the FitIns message, not the actor closure), so Ray
    # delivers the correct value to client_fn via context.run_config.
    class DriftAwareFedAvg(FedAvg):
        """FedAvg subclass that forwards the current round to client_fn."""

        def configure_fit(self, server_round, parameters, client_manager):  # type: ignore[override]
            fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
            # Inject the round number into every client's config dict.
            updated = []
            for client_proxy, fit_ins in fit_ins_list:
                new_config = dict(fit_ins.config)
                new_config["server_round"] = server_round
                from flwr.common import FitIns  # type: ignore
                updated.append((client_proxy, FitIns(fit_ins.parameters, new_config)))
            return updated

    def client_fn(context: Context) -> Client:
        c_id_int = int(context.node_config["partition-id"])

        # The round number is forwarded by DriftAwareFedAvg.configure_fit via
        # context.run_config["server_round"]. This path is serialised as part
        # of the FitIns message each round, so it crosses the Ray process
        # boundary correctly (unlike a captured mutable dict in a closure).
        if drift_simulator is not None:
            current_round = int(context.run_config.get("server_round", 0))
            current_datasets = drift_simulator.step(current_round)
            client_trainers[c_id_int].split["train"]["edge"] = (
                current_datasets[c_id_int].edges.to(device)
            )

        return FedDDIClient(
            cid=str(c_id_int),
            trainer=client_trainers[c_id_int],
            local_epochs=local_epochs,
            device=device,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    # ── Server: global encoder for server-side eval ───────────────────────────
    # Uses the same fixed node_features → feature_proj IS in the aggregation set.
    global_encoder = GraphSAGEEncoder(
        num_nodes=num_nodes,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        normalize=cfg.model.normalize,
        node_features=node_features,          # ← same fixed features
    ).to(device)

    global_predictor = NeuralLinkPredictor(
        emb_dim=cfg.model.hidden_dim,
        hidden_dim=cfg.predictor.hidden_dim,
        num_layers=cfg.predictor.num_layers,
        dropout=cfg.predictor.dropout,
        dist_dim=cfg.dataset.get("num_anchors", 0),
    ).to(device)

    val_ei = split["valid"]["edge"].t().to(device)
    full_edge_index = to_undirected(
        torch.cat([global_edge_index.to(device), val_ei], dim=1)
    )

    # ── Initial parameters (must match fl_client.py's filter exactly) ─────────
    agg_params: List[np.ndarray] = []
    for name, param in global_encoder.named_parameters():
        if _is_aggregated_encoder_param(name):
            agg_params.append(param.cpu().detach().numpy())
    for name, param in global_predictor.named_parameters():
        if _is_aggregated_predictor_param(name):
            agg_params.append(param.cpu().detach().numpy())

    initial_parameters = ndarrays_to_parameters(agg_params)

    # ── Server evaluate_fn — now produces REAL global Hits@20 ─────────────────
    server_eval_fn = build_server_evaluate_fn(
        encoder=global_encoder,
        predictor=global_predictor,
        full_edge_index=full_edge_index,
        split=split,
        evaluator=evaluator,
        device=device,
        batch_size=cfg.training.batch_size,
        node_dist_enc=dist_enc.to(device) if dist_enc is not None else None,
        using_shared_node_features=True,      # ← always True in Strategy 1
    )

    best_tracker: Dict[str, float] = {}

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = DriftAwareFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=cfg.get("fraction_evaluate", 1.0),
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn,
            fit_metrics_aggregation_fn=make_metrics_aggregation_fn("fit"),
            evaluate_metrics_aggregation_fn=make_metrics_aggregation_fn(
                "evaluate", best_tracker=best_tracker
            ),
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=num_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)

    logger.info(
        "Starting Flower simulation — Strategy 1 (Feature Bootstrapping) | "
        "shared global graph | %d clients | %d rounds | %d local epochs/round | feat_dim=%d",
        num_clients, num_rounds, local_epochs, feat_dim,
    )

    ray_gpus = 1 if (device.type == "cuda" and torch.cuda.is_available()) else 0
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config={
            "client_resources": {
                "num_cpus": 1,
                # Use the full GPU per client so Ray schedules one client at a
                # time.  The full-graph strategy means every client loads the
                # entire ogbl-ddi edge index (~4-5 GiB), so splitting the GPU
                # fraction (e.g. 1/num_clients = 0.2) causes concurrent clients
                # to collectively exceed VRAM and OOM.
                "num_gpus": 1.0 if ray_gpus > 0 else 0.0,
            }
        },
    )

    logger.info("Flower simulation finished — %d rounds completed.", num_rounds)

    if "val_hits20" in best_tracker:
        logger.info(
            "Best federated Val Hits@20 = %.4f at round %d.",
            best_tracker["val_hits20"],
            int(best_tracker["round"]),
        )
        wandb.summary["best_federated_val_hits20"] = best_tracker["val_hits20"]
        wandb.summary["best_federated_val_round"]  = int(best_tracker["round"])
    else:
        logger.warning(
            "No federated evaluation metrics recorded — check fraction_evaluate > 0."
        )

    wandb.finish()


if __name__ == "__main__":
    main()
