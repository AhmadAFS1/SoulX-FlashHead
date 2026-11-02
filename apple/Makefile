# Run from repository root: make -f apple/Makefile <target>
PYTHON ?= .venv-coreml/bin/python
export PYTHONPATH := apple
OUTPUT ?= artifacts/coreml/pro320
DEVICE_OUTPUT ?= artifacts/coreml/device/pro320
HEIGHT ?= 320
WIDTH ?= 176

.PHONY: setup fetch test export bundle app
setup:
	uv venv --python 3.12 .venv-coreml
	uv pip install --python $(PYTHON) -r apple/requirements.txt
fetch:
	$(PYTHON) -m flashhead_coreml.fetch_models
test:
	$(PYTHON) -m pytest apple/tests -q
export:
	$(PYTHON) -m flashhead_coreml.export_dit --height $(HEIGHT) --width $(WIDTH) --output $(OUTPUT) --resume
	$(PYTHON) -m flashhead_coreml.export_support --height $(HEIGHT) --width $(WIDTH) --output $(OUTPUT)
bundle:
	$(PYTHON) -m flashhead_coreml.bundle --input $(OUTPUT) --output $(DEVICE_OUTPUT) --compile
app:
	xcodegen generate --spec ios/SoulXStudio/project.yml
	xcodebuild -project ios/SoulXStudio/SoulXStudio.xcodeproj -scheme SoulXStudio -configuration Release -sdk iphoneos -allowProvisioningUpdates build
