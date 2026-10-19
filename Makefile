PYTHON ?= python3
DATA_ROOT ?= ./dataset

.PHONY: install model-v1 verifier-v1 validate test smoke smoke-strict determinism plate-mask-control check

install:
	$(PYTHON) -m pip install -r requirements.txt

model-v1:
	./scripts/fetch_model_v1.sh artifacts/models/final

verifier-v1:
	./scripts/fetch_verifier_v1.sh artifacts/models/verifier_f0

validate:
	$(PYTHON) -m reid.eval.validate_outputs --self-test

test:
	$(PYTHON) -m pytest -q

# Runs the complete pipeline on 25 query + 25 gallery images and validates all
# three output files. The installed D082 bundle (including verifier) is
# preferred over historical fusion; MODEL/VERIFIER select another stack explicitly. Without any local
# model artifact this developer check reports SKIP; use smoke-strict for release.
smoke:
	$(PYTHON) -m reid.infer.smoke --data-root "$(DATA_ROOT)" --n-images 50 $(if $(MODEL),--model "$(MODEL)") $(if $(VERIFIER),--verifier "$(VERIFIER)")

# CI/release mode: skipping because B2.1 is absent is an error.
smoke-strict:
	$(PYTHON) -m reid.infer.smoke --data-root "$(DATA_ROOT)" --n-images 50 --require-runner $(if $(MODEL),--model "$(MODEL)") $(if $(VERIFIER),--verifier "$(VERIFIER)")

# Full two-run check. Override MODEL/VERIFIER/DEVICE for another release or host.
determinism:
	PYTHONPATH=. $(PYTHON) scripts/check_determinism.py \
		--images "$(DATA_ROOT)/images" \
		--query "$(DATA_ROOT)/test_query.csv" \
		--gallery "$(DATA_ROOT)/test_gallery.csv" \
		--model "$(if $(MODEL),$(MODEL),artifacts/models/fusion)" \
		--verifier "$(if $(VERIFIER),$(VERIFIER),artifacts/models/fusion/verifier.pt)" \
		--device "$${DEVICE:-cuda}" \
		--out "$${OUT:-outputs/determinism}"

# Control run for §2.3: both the embedder and verifier see a solid plate-area fill.
plate-mask-control:
	PYTHONPATH=. $(PYTHON) scripts/run_plate_mask_control.py \
		--images "$(DATA_ROOT)/images" \
		--query "$(DATA_ROOT)/test_query.csv" \
		--gallery "$(DATA_ROOT)/test_gallery.csv" \
		--model "$(if $(MODEL),$(MODEL),artifacts/models/fusion)" \
		--verifier "$(if $(VERIFIER),$(VERIFIER),artifacts/models/fusion/verifier.pt)" \
		--device "$${DEVICE:-cuda}" \
		--out "$${OUT:-outputs/plate-mask-control}" \
		--report "$${REPORT:-outputs/plate-mask-control-report.json}"

check:
	$(MAKE) validate PYTHON="$(PYTHON)"
	$(MAKE) test PYTHON="$(PYTHON)"
	$(MAKE) smoke PYTHON="$(PYTHON)" DATA_ROOT="$(DATA_ROOT)"
