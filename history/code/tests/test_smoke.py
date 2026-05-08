import pytest


def test_torch_import():
    try:
        import torch
        assert torch.__version__ >= "2.0.0"
    except ImportError:
        pytest.fail("torch import failed")


def test_transformers_import():
    try:
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        pytest.fail("transformers import failed")


def test_numpy_import():
    try:
        import numpy
        assert numpy.__version__ >= "1.24.0"
    except ImportError:
        pytest.fail("numpy import failed")


def test_scipy_import():
    try:
        import scipy
        assert scipy.__version__ >= "1.10.0"
    except ImportError:
        pytest.fail("scipy import failed")


def test_sklearn_import():
    try:
        import sklearn
        assert sklearn.__version__ >= "1.3.0"
    except ImportError:
        pytest.fail("sklearn import failed")


def test_common_types_import():
    try:
        from common.types import SignalOutput, ModelOutput, Sample, MMQResult
    except ImportError:
        pytest.fail("common.types import failed")


def test_common_model_loader_import():
    try:
        from common.model_loader import load_smollm, load_model, get_model_output
    except ImportError:
        pytest.fail("common.model_loader import failed")


def test_common_dataset_import():
    try:
        from common.dataset import save_samples, load_samples, generate_checksum
    except ImportError:
        pytest.fail("common.dataset import failed")