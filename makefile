.PHONY: test build

build:
	docker build --no-cache --force-rm -t tnpy .

run:
	docker run --rm -i -t tnpy

test:
	python -m unittest discover -s test -p 'test_*.py'

rstdoc:
	sphinx-apidoc -o docs/source/ tnpy/

doc:
	make -C docs html
