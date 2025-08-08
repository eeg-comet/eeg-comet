"""Microstate labeling using an ONNX model (EEG-COMET)."""

import io
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import mne
import numpy as np
import onnxruntime as ort
import pandas as pd

# Silence TensorFlow warnings before importing ONNX Runtime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hide INFO and WARNING messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.*")



class MicrostateLabeler:
    """Label microstate maps using a pre-trained ONNX model."""

    def __init__(self, microstate_maps, eeg_info, microstate_maps_path):
        """Initialize the microstate labeler.

        Args:
            microstate_maps (ndarray): Microstate maps to label.
            eeg_info (dict): EEG info (e.g., MNE info or channel metadata).
            microstate_maps_path (str): Output path for labeled maps.
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

    def do_labeling(self):
        """Perform microstate labeling using a trained model."""
        # Match the size used during training
        image_size = 256
        images = []

        for i in range(self.microstate_maps.shape[0]):
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(
                self.microstate_maps[i, :],
                self.eeg_info,
                contours=10,
                sensors=False,
                axes=ax,
                show=False,
                sphere="auto",
            )
            with io.BytesIO() as buf:
                fig.savefig(buf, dpi=200, bbox_inches="tight")
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            plt.close(fig)  # Close the figure to free memory

            image = cv2.imdecode(img_arr, 1)
            image = cv2.resize(image, (image_size, image_size))

            # Convert BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert from HWC to CHW format (from [H,W,C] to [C,H,W])
            image = image.transpose(2, 0, 1)

            # Apply normalization as in training
            image = image / 255.0  # Scale to [0,1]
            mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
            image = (image - mean) / std

            image = np.expand_dims(image, axis=0)  # Add batch dimension
            images.append(image)

        stacked_microstate_images = np.vstack(images)

        # Load ONNX model and do inference
        # The exported classifier has 7 output neurons (letters A–G).
        num_classes = 7  # fixed – do not change
        assert self.n_states <= num_classes, (
            f"The labeling model can assign at most {num_classes} unique labels, "
            f"but {self.n_states} microstate maps were provided."
        )

        # Dictionary that maps class index → character label (0→'A', 1→'B', …, 6→'G')
        dictionary2use = {i: chr(ord("A") + i) for i in range(num_classes)}

        # Update model path to match the exported ONNX model
        model_path = "./models/model_v1.3.onnx"

        # Initialize ONNX Runtime session
        session = ort.InferenceSession(model_path)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        predictions = session.run(
            [output_name], {input_name: stacked_microstate_images.astype(np.float32)}
        )[0]

        softmax_predictions = self.softmax(predictions) * 100
        assigned_labels, probabilities = self.get_labels(
            predictions, softmax_predictions, dictionary2use
        )
        overall_confidence = sum(probabilities.values()) / len(probabilities)

        # Build the final ordered list of labels, guaranteeing uniqueness.
        micro_labels = []
        available_chars = [chr(ord("A") + i) for i in range(num_classes)]

        for i in range(self.n_states):
            if i in assigned_labels:
                micro_labels.append(assigned_labels[i])
            else:
                # Fallback – pick the first unused character from A–G
                for ch in available_chars:
                    if ch not in micro_labels:
                        micro_labels.append(ch)
                        break

        self.micro_labels = micro_labels

        # Save Best Maps
        maps_df = pd.DataFrame(
            self.microstate_maps.T, columns=micro_labels, index=self.eeg_info["ch_names"]
        )
        maps_df.to_csv(self.microstate_maps_path)
        return micro_labels, overall_confidence

    @staticmethod
    def get_labels(confidences, softmax_predictions, dictionary2use):
        """Assign a unique label from A–G to each microstate.

        A greedy approach is used:
        For each microstate (row) we take its class probabilities, select the
        highest-scoring label that has not yet been assigned to any other
        microstate, and record the associated confidence.
        This works because the number of microstates (rows) is guaranteed to
        be ≤ 7 – the total number of available labels.
        """
        assigned_labels = {}
        probabilities = {}
        used_chars = set()

        for image_index in range(confidences.shape[0]):
            # Sort class scores for this map in descending order
            sorted_label_indices = np.argsort(confidences[image_index])[::-1]

            for lbl_idx in sorted_label_indices:
                label_char = dictionary2use.get(lbl_idx, chr(ord("A") + lbl_idx))

                if label_char not in used_chars:
                    assigned_labels[image_index] = label_char
                    probabilities[image_index] = softmax_predictions[image_index, lbl_idx]
                    used_chars.add(label_char)
                    break  # Move to next microstate once a unique label is assigned

        return assigned_labels, probabilities
