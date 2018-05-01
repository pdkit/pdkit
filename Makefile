.PHONY: help clean package dev

help:
	@echo "This project assumes that an active Python virtualenv is present."
	@echo "The following make targets are available:"
	@echo "  package    create package to upload to pypi"
	@echo "  dev	    install all relveant modules"

clean:
	rm -rf dist/*

dev:
	pip install -r requirements.txt
	pip install --upgrade pip wheel setuptools twine
	pip install git+https://github.com/blue-yonder/tsfresh
	pip install -e .
	pip freeze

package:
	python setup.py sdist
	python setup.py bdist_wheel --universal