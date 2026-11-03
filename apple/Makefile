# Run from repository root: make -f apple/Makefile <target>
PYTHON ?= .venv-coreml/bin/python
export PYTHONPATH := apple
OUTPUT ?= artifacts/coreml/pro320
DEVICE_OUTPUT ?= artifacts/coreml/device/pro320
HEIGHT ?= 320
WIDTH ?= 176
# Transformer variant: baseline, fold, ffnk2+fold, ffnmap0=0.5,...+fold (see flashhead_coreml/variants.py)
VARIANT ?= baseline

.PHONY: setup setup-cpu-ab fetch test export bundle app
setup:
	uv venv --python 3.12 .venv-coreml
	uv pip install --python $(PYTHON) -r apple/requirements.txt
setup-cpu-ab: setup
	uv pip install --python $(PYTHON) -r apple/requirements-cpu-ab.txt
fetch:
	$(PYTHON) -m flashhead_coreml.fetch_models
test:
	$(PYTHON) -m pytest apple/tests -q
export:
	$(PYTHON) -m flashhead_coreml.export_dit --height $(HEIGHT) --width $(WIDTH) --output $(OUTPUT) --variant '$(VARIANT)' --resume
	$(PYTHON) -m flashhead_coreml.export_support --height $(HEIGHT) --width $(WIDTH) --output $(OUTPUT)
bundle:
	$(PYTHON) -m flashhead_coreml.bundle --input $(OUTPUT) --output $(DEVICE_OUTPUT) --compile
app:
	xcodegen generate --spec ios/SoulXStudio/project.yml
	xcodebuild -project ios/SoulXStudio/SoulXStudio.xcodeproj -scheme SoulXStudio -configuration Release -sdk iphoneos -allowProvisioningUpdates build
