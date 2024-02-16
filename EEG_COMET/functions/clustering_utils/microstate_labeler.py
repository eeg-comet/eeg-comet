
import mne
import io
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model


class MicrostateLabeler:
    def __init__(self, best_maps, eeg_info, microstate_maps_path):
        self.best_maps = best_maps
        self.n_states = best_maps.shape[0]
        self.eeg_info = eeg_info
        self.microstate_maps_path = microstate_maps_path
        self.micro_labels = []

    def do_labeling(self):
        image_size = 448
        images = []

        for i in range(self.best_maps.shape[0]):
            fig, ax = plt.subplots()
            mne.viz.plot_topomap(self.best_maps[i, :], self.eeg_info, contours=10, sensors=False, axes=ax, show=False,
                                 sphere='auto')
            with io.BytesIO() as buf:
                fig.savefig(buf, dpi=200, bbox_inches='tight')
                buf.seek(0)
                img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            image = cv2.imdecode(img_arr, 1)
            image = cv2.resize(image, (image_size, image_size))
            image = np.expand_dims(image, axis=0)
            images.append(image)

        image = np.vstack(images)
        print(image.shape)

        # Load model and do inference
        maps = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
        model_path = './models/model_v1.22.h5'
        model = load_model(model_path, compile=False)
        output = model.predict(image)
        label_result = self.get_labels(output, maps)

        micro_labels = []
        additional_label = 'H'
        assert self.n_states < 27, 'Cannot label more than 27 microstates: not enough letters'

        for i in range(self.n_states):
            if i in label_result:
                micro_labels.append(label_result[i])
            else:
                micro_labels.append(additional_label)
                additional_label = chr(ord(additional_label) + 1)

        self.micro_labels = micro_labels
        print(self.micro_labels)

        # Save Best Maps
        maps_df = pd.DataFrame(self.best_maps.T, columns=micro_labels, index=self.eeg_info['ch_names'])
        maps_df.to_csv(self.microstate_maps_path)
        return micro_labels

    @staticmethod
    def get_labels(confidences, maps):
        image_pool = set()
        state_pool = set()
        result = {}

        while True:
            image = confidences.argmax() // confidences.shape[1]
            state = confidences.argmax() % confidences.shape[1]

            if image not in image_pool and state not in state_pool:
                image_pool.add(image)
                state_pool.add(state)
                confidences[image, state] = np.NINF
                result[image] = maps[state]
            else:
                confidences[image, state] = np.NINF
            if len(result) == confidences.shape[0] or len(result) == 7:
                break

        print(result)
        return result
