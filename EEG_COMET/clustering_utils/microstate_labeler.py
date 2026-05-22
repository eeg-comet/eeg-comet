"""Microstate labeling using an ONNX model (EEG-COMET).

Uses fast scipy interpolation of raw channel values to 128x128
topography arrays for model inference (no matplotlib rendering needed).
"""

from pathlib import Path

import mne
import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.spatial import Delaunay, cKDTree

# Resolve the ONNX model path relative to this file so the labeler works
# regardless of the current working directory. Falls back to a CWD-relative
# location when the packaged model file is not present.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PACKAGE_ROOT / "models" / "model_v2.onnx"
_FALLBACK_MODEL_PATH = Path("./models/model_v2.onnx")
_MODEL_PATH = str(
    _DEFAULT_MODEL_PATH if _DEFAULT_MODEL_PATH.exists() else _FALLBACK_MODEL_PATH
)
_IMAGE_SIZE = 128


class FastTopographyInterpolator:
    """Convert EEG channel values to 2D topography images via pre-computed
    interpolation weights (Delaunay triangulation + barycentric coordinates).

    The weight matrix is computed once at init; each subsequent call is a
    single matrix-vector multiply, making this ~100x faster than rendering
    through matplotlib.
    """

    def __init__(self, info, image_size=_IMAGE_SIZE):
        self.image_size = image_size

        picks = mne.pick_types(info, eeg=True, exclude='bads')
        self.n_channels = len(picks)

        try:
            pos = mne.channels.layout._find_topomap_coords(info, picks=picks)
        except Exception:
            pos = np.array([info['chs'][p]['loc'][:2] for p in picks])

        pos = pos - pos.mean(axis=0)
        max_range = np.abs(pos).max()
        if max_range > 0:
            pos = pos / (max_range * 1.1)
        self.pos = pos

        x = np.linspace(-1, 1, image_size)
        y = np.linspace(-1, 1, image_size)
        self.grid_x, self.grid_y = np.meshgrid(x, y)

        self.head_radius = 1.0
        self.mask = np.sqrt(self.grid_x ** 2 + self.grid_y ** 2) <= self.head_radius
        self.mask_flat = self.mask.ravel()

        self._precompute_interpolation_weights()
        self._precompute_colormap_lut()
        self._precompute_background()

    def _precompute_interpolation_weights(self):
        grid_points = np.column_stack([self.grid_x.ravel(), self.grid_y.ravel()])
        n_grid = len(grid_points)

        tri = Delaunay(self.pos)
        simplex_indices = tri.find_simplex(grid_points)

        weights = np.zeros((n_grid, self.n_channels), dtype=np.float64)

        inside_mask = simplex_indices >= 0
        inside_idx = np.where(inside_mask)[0]

        if len(inside_idx) > 0:
            si = simplex_indices[inside_idx]
            vertices = tri.simplices[si]
            T = tri.transform[si]
            delta = grid_points[inside_idx] - T[:, 2]
            bary_2 = np.einsum('ijk,ik->ij', T[:, :2], delta)
            bary_3 = 1.0 - bary_2.sum(axis=1)
            bary = np.column_stack([bary_2, bary_3])
            for k in range(3):
                weights[inside_idx, vertices[:, k]] = bary[:, k]

        outside_idx = np.where(~inside_mask)[0]
        if len(outside_idx) > 0:
            tree = cKDTree(self.pos)
            _, nearest = tree.query(grid_points[outside_idx])
            weights[outside_idx, nearest] = 1.0

        weights[~self.mask_flat] = 0.0
        self.weight_matrix = weights.astype(np.float32)

    def _precompute_colormap_lut(self, n_levels=256):
        t = np.linspace(0.0, 1.0, n_levels).astype(np.float32)
        lut = np.zeros((n_levels, 3), dtype=np.float32)
        lut[:, 0] = np.where(t > 0.5, 0.7 + 0.3 * (t - 0.5) / 0.5,
                             0.7 - 0.5 * (0.5 - t) / 0.5)
        lut[:, 1] = 1.0 - 2.0 * np.abs(t - 0.5)
        lut[:, 2] = np.where(t < 0.5, 0.7 + 0.6 * (0.5 - t) / 0.5,
                             0.7 - 0.6 * (t - 0.5) / 0.5)
        lut = np.clip(lut, 0.0, 1.0)
        self.colormap_lut = (lut * 255).astype(np.uint8)

    def _precompute_background(self):
        self.inside_indices = np.where(self.mask_flat)[0]
        self.bg_rgb = np.full((self.image_size * self.image_size, 3), 255, dtype=np.uint8)

    def to_rgb_image(self, eeg_values):
        """Convert ``(n_channels,)`` EEG values to a ``(H, W, 3)`` uint8 RGB image."""
        grid_flat = self.weight_matrix @ eeg_values.astype(np.float32)
        values = grid_flat[self.inside_indices]

        absmax = np.abs(values).max()
        if absmax > 0:
            values = values / absmax

        lut_idx = ((values + 1.0) * 127.5).astype(np.intp)
        np.clip(lut_idx, 0, 255, out=lut_idx)

        rgb_flat = self.bg_rgb.copy()
        rgb_flat[self.inside_indices] = self.colormap_lut[lut_idx]
        return rgb_flat.reshape(self.image_size, self.image_size, 3)


class MicrostateLabeler:
    """Label microstate maps using the pre-trained v2 ONNX model."""

    def __init__(self, microstate_maps, eeg_info, microstate_maps_path):
        """Initialize the microstate labeler.

        Args:
            microstate_maps (ndarray): Microstate maps to label, shape (n_states, n_channels).
            eeg_info: MNE Info object with channel positions.
            microstate_maps_path (str): Output path for labeled maps CSV.
        """
        self.microstate_maps = microstate_maps
        self.n_states = microstate_maps.shape[0]
        self.eeg_info = eeg_info
        self.microstate_maps_path = microstate_maps_path
        self.micro_labels = []

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    @staticmethod
    def _normalize_hwc(image_hwc):
        """HWC uint8 RGB image -> (1, 3, H, W) float32 with ImageNet normalization."""
        image = image_hwc.transpose(2, 0, 1).astype(np.float64) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
        image = (image - mean) / std
        return np.expand_dims(image, axis=0)

    def _generate_images(self):
        """Fast scipy interpolation to 128x128 topography images."""
        interpolator = FastTopographyInterpolator(self.eeg_info, _IMAGE_SIZE)
        images = []
        for i in range(self.n_states):
            rgb = interpolator.to_rgb_image(self.microstate_maps[i, :])
            images.append(self._normalize_hwc(rgb))
        return images

    def do_labeling(self):
        """Perform microstate labeling using the trained model."""
        num_classes = 7
        assert self.n_states <= num_classes, (
            f"The labeling model can assign at most {num_classes} unique labels, "
            f"but {self.n_states} microstate maps were provided."
        )

        images = self._generate_images()

        dictionary2use = {i: chr(ord("A") + i) for i in range(num_classes)}

        session = ort.InferenceSession(_MODEL_PATH)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        all_predictions = []
        for img in images:
            pred = session.run(
                [output_name], {input_name: img.astype(np.float32)}
            )[0]
            all_predictions.append(pred)
        predictions = np.vstack(all_predictions)

        softmax_predictions = self.softmax(predictions) * 100
        assigned_labels, probabilities = self.get_labels(
            predictions, softmax_predictions, dictionary2use
        )
        overall_confidence = sum(probabilities.values()) / len(probabilities)

        micro_labels = []
        available_chars = [chr(ord("A") + i) for i in range(num_classes)]

        for i in range(self.n_states):
            if i in assigned_labels:
                micro_labels.append(assigned_labels[i])
            else:
                for ch in available_chars:
                    if ch not in micro_labels:
                        micro_labels.append(ch)
                        break

        self.micro_labels = micro_labels

        label_confidences = {}
        for i in range(self.n_states):
            label = micro_labels[i]
            if i in probabilities:
                label_confidences[label] = probabilities[i]
            else:
                label_confidences[label] = 0.0

        maps_df = pd.DataFrame(
            self.microstate_maps.T, columns=micro_labels, index=self.eeg_info["ch_names"]
        )
        maps_df.to_csv(self.microstate_maps_path)
        return micro_labels, overall_confidence, label_confidences

    @staticmethod
    def get_labels(confidences, softmax_predictions, dictionary2use):
        """Assign a unique label from A-G to each microstate.

        A greedy approach is used:
        For each microstate (row) we take its class probabilities, select the
        highest-scoring label that has not yet been assigned to any other
        microstate, and record the associated confidence.
        This works because the number of microstates (rows) is guaranteed to
        be <= 7 -- the total number of available labels.
        """
        assigned_labels = {}
        probabilities = {}
        used_chars = set()

        for image_index in range(confidences.shape[0]):
            sorted_label_indices = np.argsort(confidences[image_index])[::-1]

            for lbl_idx in sorted_label_indices:
                label_char = dictionary2use.get(lbl_idx, chr(ord("A") + lbl_idx))

                if label_char not in used_chars:
                    assigned_labels[image_index] = label_char
                    probabilities[image_index] = softmax_predictions[image_index, lbl_idx]
                    used_chars.add(label_char)
                    break

        return assigned_labels, probabilities
