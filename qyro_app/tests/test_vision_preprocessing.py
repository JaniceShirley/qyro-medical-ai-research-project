import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from qyro_app.config import REPO_ROOT
from qyro_app.vision import VisionModels

sys.path.append(str(REPO_ROOT))


def test_app_preprocessing_matches_training_val_transform():
    """The app must feed classifiers exactly what training/evaluation fed them."""
    from scripts.training.train_severity import get_dataloaders

    rng = np.random.default_rng(3)
    image_bgr = rng.integers(0, 256, (1024, 768, 3), dtype=np.uint8)
    path = os.path.join(os.path.dirname(__file__), "_tmp_preproc.png")
    cv2.imwrite(path, image_bgr)
    try:
        # get_dataloaders builds all three loaders, so every split needs a row
        df = pd.DataFrame({"image_path": [path] * 3, "split": ["train", "valid", "test"], "severity_label": ["stage_1"] * 3})
        _, _, test_loader = get_dataloaders(df, 1, 0, False, None)
        test_loader.dataset.workspace_root = ""
        expected, _ = next(iter(test_loader))
    finally:
        os.remove(path)

    vm = VisionModels()
    vm.device = "cpu"
    actual = vm._classifier_input(image_bgr)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, atol=1e-5)
