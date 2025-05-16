import mne
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnxruntime as ort


class MicrostateLabeler:
    def __init__(self, microstate_maps, eeg_info, microstate_maps_path):
        """
        Class for performing microstate labeling using a trained model.

        Args:
            microstate_maps (ndarray): The microstate maps.
            eeg_info (dict): Information about the EEG.
            microstate_maps_path (str): The path to save the microstate maps.
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
        image_size = 448
        images = []

        for i in range(self.microstate_maps.shape[0]):
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(
                self.microstate_maps[i, :], self.eeg_info,
                contours=10, sensors=False, axes=ax, show=False, sphere='auto'
            )
            with io.BytesIO() as buf:
                fig.savefig(buf, dpi=200, bbox_inches='tight')
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(img_arr, 1)
            image = cv2.resize(image, (image_size, image_size))
            image = np.expand_dims(image, axis=0)
            images.append(image)
        stacked_microstate_images = np.vstack(images)

        # Load ONNX model and do inference
        num_classes = self.microstate_maps.shape[0]
        dictionary2use = {i: chr(ord('A') + i) for i in range(num_classes)}

        # Change model path to the ONNX model
        model_path = './models/model_v1.22.onnx'

        # Initialize ONNX Runtime session
        session = ort.InferenceSession(model_path)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Run inference
        predictions = session.run([output_name], {input_name: stacked_microstate_images.astype(np.float32)})[0]

        softmax_predictions = self.softmax(predictions) * 100
        assigned_labels, probabilities = self.get_labels(predictions, softmax_predictions, dictionary2use)
        overall_confidence = sum(probabilities.values()) / len(probabilities)

        micro_labels = []
        additional_label = chr(ord('A') + 7)
        assert self.n_states < 27, 'Cannot label more than 27 microstates: not enough letters'

        for i in range(self.n_states):
            if i in assigned_labels:
                micro_labels.append(assigned_labels[i])
            else:
                micro_labels.append(additional_label)
                additional_label = chr(ord(additional_label) + 1)

        self.micro_labels = micro_labels

        # Save Best Maps
        maps_df = pd.DataFrame(self.microstate_maps.T, columns=micro_labels, index=self.eeg_info['ch_names'])
        maps_df.to_csv(self.microstate_maps_path)
        return micro_labels, overall_confidence

    @staticmethod
    def get_labels(confidences, softmax_predictions, dictionary2use):
        """Get microstate labels based on confidence scores."""
        assigned_labels = {}
        probabilities = {}

        num_classes = len(dictionary2use)
        label_indices = list(range(num_classes))
        label_characters = [dictionary2use[i] if i < num_classes else chr(ord('H') + i - num_classes) for i in
                            label_indices]

        for label_index, label_char in zip(label_indices, label_characters):
            max_confidence = -np.inf
            max_image_index = -1

            for image_index in range(confidences.shape[0]):
                if image_index not in assigned_labels:
                    if label_index < confidences.shape[1]:  # Ensure label index is within bounds
                        confidence = confidences[image_index, label_index]
                        if confidence > max_confidence:
                            max_confidence = confidence
                            max_image_index = image_index

            if max_image_index != -1:
                assigned_labels[max_image_index] = label_char
                probabilities[max_image_index] = softmax_predictions[max_image_index, label_index]

        return assigned_labels, probabilities
