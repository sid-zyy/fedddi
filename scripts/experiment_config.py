"""Build Hydra overrides for reproducible experiment runs."""
from typing import Any, Dict, List
import json

def build_overrides(
    exp: Dict[str, Any],
    defaults: Dict[str, Any],
    poison_specs: Dict[str, Any],
    datadrift_specs: Dict[str, Any],
    csv_output_dir: str
) -> List[str]:
    overrides = [
        f"num_clients={defaults['num_clients']}",
        f"num_rounds={exp['num_rounds']}",
        f"local_epochs={defaults['local_epochs']}",
        f"local_observations.dirichlet_alpha={defaults['dirichlet_alpha']}",
        f"training.seed={exp['seed']}",
        f"fl.algorithm={exp['algorithm']}",
        f'device=cuda',
        f'wandb.mode=disabled',
        f'+csv_output_dir={csv_output_dir}'
    ]
    if exp['algorithm'] == 'fedprox':
        overrides.append(f"fedprox.mu={exp['fedprox_mu']}")
    poison_events = poison_specs.get(exp['poison'], [])
    if poison_events:
        events_str = json.dumps(poison_events).replace('"', '')
        overrides.append(f'poisoning.events={events_str}')
    datadrift_events = datadrift_specs.get(exp['datadrift'], [])
    if datadrift_events:
        events_str = json.dumps(datadrift_events).replace('"', '')
        overrides.append(f'data_drift.events={events_str}')
    defense = exp.get('defense')
    if defense:
        overrides.append(f"defense.mode={defense.get('mode', 'none')}")
        for key, value in defense.items():
            if key == 'mode':
                continue
            if value is None:
                continue
            overrides.append(f'defense.{key}={value}')
    return overrides
