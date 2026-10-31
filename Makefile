CONFIG ?= configs/experiment.yaml
SPLIT  ?= fixed
SEED   ?= 0

setup:
	pip install -e ".[generate,dev]"
test:
	pytest -q
real:
	python -m tccdet.collect_real --config $(CONFIG)
generate:   # make generate G=sd15
	python -m tccdet.generate.run --config $(CONFIG) --generator $(G)
preprocess:
	python -m tccdet.preprocess --config $(CONFIG)
split:
	python -m tccdet.split --config $(CONFIG)
shortcuts:
	python -m tccdet.shortcuts --config $(CONFIG) --split $(SPLIT)
train:
	python -m tccdet.train --config $(CONFIG) --split $(SPLIT) --seed $(SEED)
eval:
	python -m tccdet.evaluate --config $(CONFIG) --split $(SPLIT) --seed $(SEED)
aggregate:
	python -m tccdet.aggregate --config $(CONFIG) --split $(SPLIT)
smoke:      # pipeline completo com dados falsos (não usa GPU)
	python scripts/make_fake_data.py --config configs/smoke.yaml
	$(MAKE) preprocess split shortcuts train eval CONFIG=configs/smoke.yaml
all: preprocess split shortcuts train eval
