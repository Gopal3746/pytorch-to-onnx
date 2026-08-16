.PHONY: setup test smoke benchmark compile-benchmark clean

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

smoke:
	python benchmark.py --backend pytorch --weights random --batch-sizes 1 --warmup 1 --iterations 2

benchmark:
	python benchmark.py --backend all --batch-sizes 1 8 32 --warmup 10 --iterations 50 --threads 1

compile-benchmark:
	python benchmark.py --backend torchcompile --batch-sizes 1 8 32 --warmup 10 --iterations 50 --threads 1

clean:
	rm -rf artifacts/results artifacts/*.onnx .pytest_cache
