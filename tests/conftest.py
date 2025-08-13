import os
import random

import numpy as np
import pytest
import torch
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _set_reproducible_env():
    # Make tests deterministic
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    yield
