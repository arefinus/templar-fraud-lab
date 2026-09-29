PYTHON ?= python

.PHONY: setup smoke demo test evaluate train figures audit fixture clean

setup:
	$(PYTHON) -m pip install -e . && $(PYTHON) -m pip install pytest

smoke:
	$(PYTHON) -m templar_fraud_lab smoke

demo:
	$(PYTHON) -m templar_fraud_lab demo

test:
	$(PYTHON) -m pytest -q

evaluate:
	$(PYTHON) -m templar_fraud_lab evaluate

train:
	$(PYTHON) -m templar_fraud_lab train

figures:
	$(PYTHON) tools/render_figures.py && $(PYTHON) -m templar_fraud_lab figures

audit:
	$(PYTHON) -m templar_fraud_lab audit

fixture:
	$(PYTHON) tools/make_fixture.py

clean:
	rm -rf .smoke_output .pytest_cache
