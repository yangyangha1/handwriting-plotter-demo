.PHONY: install install-ocr test lint compile gui clean

install:
	python -m pip install -e '.[dev]'

install-ocr:
	python -m pip install -e '.[ocr]'

test:
	pytest -q

lint:
	ruff check src tests

compile:
	python -m compileall -q src

gui:
	python -m hwplotter.gui

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info out
