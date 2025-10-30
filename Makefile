.PHONY: help clean package dev test

help:
	@echo "This project assumes that an active Python virtualenv is present."
	@echo "The following make targets are available:"
	@echo "  package    create package to upload to pypi"
	@echo "  dev	    install all relevant modules"
	@echo "  test	    test all relevant modules"

clean:
	rm -rf dist/*

dev:
	pip install -r requirements.txt
	pip install --upgrade pip wheel setuptools twine
	pip install -e .
	pip freeze

test:
	python tests/test_tremor_processing.py
	python tests/test_gait_processing.py
	python tests/test_bradykinesia_processing.py
	python tests/test_finger_tapping_processing.py
	python tests/test_result_set.py
	python tests/test_voice_processor.py

package:
	python setup.py sdist
	python setup.py bdist_wheel --universal