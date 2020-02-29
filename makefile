.PHONY: test build

build:
	docker build --no-cache --force-rm -t tnpy .

run:
	docker run --rm -it --entrypoint=/bin/bash -v ${PWD}:/home/project tnpy -i

test:
	python -m unittest discover -s test -p '*_test.py'

rstdoc:
	sphinx-apidoc -o docs/source/ lib/

doc:
	make -C docs html
