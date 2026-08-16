import pytest

from inference_bench.cli import positive_int, selected_backends


def test_selected_backends_all():
    assert selected_backends("all") == ["pytorch", "onnxruntime"]
    assert selected_backends("all", include_compile=True) == ["pytorch", "torchcompile", "onnxruntime"]


def test_positive_int_rejects_zero():
    with pytest.raises(Exception):
        positive_int("0")
