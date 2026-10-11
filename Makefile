PYTHON ?= python

.PHONY: install prepare-data train train-fl smoke plan test test-fast test-caga

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

prepare-data:
	$(PYTHON) -c "from src.data.dataset import load_dataset; load_dataset(root='./data')"
	$(PYTHON) scripts/prepare_atc_mapping.py

train:
	$(PYTHON) train.py baseline.use_fixed_features=true wandb.mode=disabled

train-fl:
	$(PYTHON) main_fl.py defense.mode=caga wandb.mode=disabled

smoke:
	$(PYTHON) main_fl.py defense.mode=caga num_clients=3 num_rounds=2 local_epochs=1 wandb.mode=disabled plotting.enabled=false

plan:
	$(PYTHON) scripts/run_experiments.py --tiers 1 --seeds 0

test:
	$(PYTHON) -m pytest tests/

test-fast:
	$(PYTHON) -m pytest tests/ -m "not slow and not integration"

test-caga:
	$(PYTHON) -m pytest tests/test_caga.py tests/test_caga_sim.py tests/test_actor_cache.py tests/test_journal_protocol.py tests/test_caga_strategy.py
