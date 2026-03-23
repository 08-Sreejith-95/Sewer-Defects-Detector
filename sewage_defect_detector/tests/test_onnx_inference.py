"""
Unit tests for onnx_inference.py.
No real dataset or model file required — uses synthetic images
and a mocked ONNX session throughout.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image

from onnx_inference import (
    preprocess,
    predict_single,
    CLASS_NAMES,
    IMG_SIZE,
    MEAN,
    STD,
    THRESHOLD,
)

NUM_CLASSES = len(CLASS_NAMES)  # 19


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_jpg(tmp_path):
    """Random 640x480 RGB JPEG — realistic phone-camera dimensions."""
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    p = tmp_path / "frame_001.jpg"
    Image.fromarray(arr).save(p)
    return str(p)

@pytest.fixture
def black_png(tmp_path):
    """All-black PNG — edge case for normalisation."""
    p = tmp_path / "black.png"
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(p)
    return str(p)

@pytest.fixture
def white_png(tmp_path):
    """All-white PNG — opposite normalisation edge case."""
    p = tmp_path / "white.png"
    Image.fromarray(np.full((256, 256, 3), 255, dtype=np.uint8)).save(p)
    return str(p)

@pytest.fixture
def mock_session():
    """Mock ORT session returning random logits — no .onnx file needed."""
    sess = MagicMock()
    sess.run.return_value = [
        np.random.randn(1, NUM_CLASSES).astype(np.float32)
    ]
    return sess

@pytest.fixture
def positive_session():
    """Mock session where all logits are very high → all classes fire."""
    sess = MagicMock()
    sess.run.return_value = [
        np.full((1, NUM_CLASSES), 10.0, dtype=np.float32)
    ]
    return sess

@pytest.fixture
def negative_session():
    """Mock session where all logits are very low → no class fires."""
    sess = MagicMock()
    sess.run.return_value = [
        np.full((1, NUM_CLASSES), -10.0, dtype=np.float32)
    ]
    return sess


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocess:

    def test_output_shape(self, sample_jpg):
        arr = preprocess(sample_jpg)
        assert arr.shape == (1, 3, IMG_SIZE, IMG_SIZE), \
            f"Expected (1,3,{IMG_SIZE},{IMG_SIZE}), got {arr.shape}"

    def test_output_dtype(self, sample_jpg):
        assert preprocess(sample_jpg).dtype == np.float32

    def test_resize_from_non_square(self, sample_jpg):
        """640x480 input must be resized to square IMG_SIZE."""
        arr = preprocess(sample_jpg)
        assert arr.shape[-2:] == (IMG_SIZE, IMG_SIZE)

    def test_black_image_normalisation(self, black_png):
        """Pixel value 0 → (0 - mean) / std — should be negative."""
        arr = preprocess(black_png)
        assert arr.mean() < 0, "Black image should normalise to negative values"

    def test_white_image_normalisation(self, white_png):
        """Pixel value 255 → (1.0 - mean) / std — should be positive."""
        arr = preprocess(white_png)
        assert arr.mean() > 0, "White image should normalise to positive values"

    def test_channel_order_is_rgb(self, tmp_path):
        """Verify R, G, B channels are normalised with their respective stats."""
        # Create image with R=255, G=0, B=0
        arr_img = np.zeros((64, 64, 3), dtype=np.uint8)
        arr_img[:, :, 0] = 255
        p = tmp_path / "red.png"
        Image.fromarray(arr_img).save(p)
        out = preprocess(str(p))
        # R channel (idx 0): (1.0 - 0.485) / 0.229 ≈ 2.25
        # G channel (idx 1): (0.0 - 0.456) / 0.224 ≈ -2.04
        assert out[0, 0].mean() > 1.0,  "R channel should be positive for red image"
        assert out[0, 1].mean() < -1.0, "G channel should be negative for red image"

    def test_handles_png(self, black_png):
        arr = preprocess(black_png)
        assert arr.shape == (1, 3, IMG_SIZE, IMG_SIZE)


# ── Prediction ────────────────────────────────────────────────────────────────

class TestPredictSingle:

    def test_returns_required_keys(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        assert {"image", "labels", "probabilities", "latency_ms"} \
               .issubset(result.keys())

    def test_image_path_preserved(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        assert result["image"] == sample_jpg

    def test_labels_are_subset_of_class_names(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        for label in result["labels"]:
            assert label in CLASS_NAMES, f"Unknown label returned: {label}"

    def test_probabilities_keys_match_class_names(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        assert set(result["probabilities"].keys()) == set(CLASS_NAMES)

    def test_probabilities_in_unit_interval(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        for cls, p in result["probabilities"].items():
            assert 0.0 <= p <= 1.0, f"{cls}: probability {p} outside [0, 1]"

    def test_latency_positive(self, sample_jpg, mock_session):
        result = predict_single(mock_session, sample_jpg)
        assert result["latency_ms"] > 0

    def test_all_logits_high_fires_all_classes(self, sample_jpg, positive_session):
        result = predict_single(positive_session, sample_jpg, threshold=0.4)
        assert set(result["labels"]) == set(CLASS_NAMES)

    def test_all_logits_low_returns_ok(self, sample_jpg, negative_session):
        result = predict_single(negative_session, sample_jpg, threshold=0.4)
        assert result["labels"] == ["OK"]

    def test_threshold_zero_fires_all(self, sample_jpg):
        """threshold=0.0 means sigmoid of any finite logit fires."""
        sess = MagicMock()
        sess.run.return_value = [
            np.zeros((1, NUM_CLASSES), dtype=np.float32)  # sigmoid(0) = 0.5
        ]
        result = predict_single(sess, sample_jpg, threshold=0.0)
        assert len(result["labels"]) == NUM_CLASSES

    def test_threshold_one_fires_none(self, sample_jpg):
        """threshold=1.0 is unreachable by sigmoid — fallback to OK."""
        sess = MagicMock()
        sess.run.return_value = [
            np.zeros((1, NUM_CLASSES), dtype=np.float32)
        ]
        result = predict_single(sess, sample_jpg, threshold=1.0)
        assert result["labels"] == ["OK"]

    def test_custom_threshold_respected(self, sample_jpg):
        """Logit of 0 → prob 0.5; threshold 0.6 should suppress it."""
        sess = MagicMock()
        sess.run.return_value = [
            np.zeros((1, NUM_CLASSES), dtype=np.float32)
        ]
        result = predict_single(sess, sample_jpg, threshold=0.6)
        assert result["labels"] == ["OK"]

    def test_session_called_with_correct_input_key(self, sample_jpg, mock_session):
        predict_single(mock_session, sample_jpg)
        call_kwargs = mock_session.run.call_args
        inp_dict = call_kwargs[0][1]          # positional arg 1 = feed_dict
        assert "image" in inp_dict, "Session must be called with key 'image'"

    def test_session_receives_correct_shape(self, sample_jpg, mock_session):
        predict_single(mock_session, sample_jpg)
        inp = mock_session.run.call_args[0][1]["image"]
        assert inp.shape == (1, 3, IMG_SIZE, IMG_SIZE)


# ── Class list sanity ─────────────────────────────────────────────────────────

class TestClassNames:

    def test_count(self):
        assert NUM_CLASSES == 19

    def test_no_duplicates(self):
        assert len(CLASS_NAMES) == len(set(CLASS_NAMES))

    def test_ok_present(self):
        assert "OK" in CLASS_NAMES

    def test_expected_classes_present(self):
        for cls in ["VA", "RB", "OB", "PF", "DE", "FS", "ND"]:
            assert cls in CLASS_NAMES, f"Missing expected class: {cls}"