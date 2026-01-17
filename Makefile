# GeoGraph-3D Automation

PYTHON := python
PIP := pip

.PHONY: install train test visualize clean

install:
	$(PIP) install -r requirements.txt

train:
	$(PYTHON) train.py

test:
	$(PYTHON) -m unittest discover tests/

visualize:
	$(PYTHON) src/visualize.py

robustness:
	$(PYTHON) src/evaluate_robustness.py

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf checkpoints/