.PHONY: install test train-local docker-build docker-train-small

install:
	pip3 install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu --break-system-packages
	pip3 install -e ".[dev]" --break-system-packages

test:
	pytest tests/ -v

train-local:
	python3 -m connectzero.cli train --config $(CONFIG)

docker-build:
	docker build -f docker/Dockerfile.train -t connectzero-train .

docker-train-small:
	docker run --rm -v $(PWD)/runs:/app/runs connectzero-train