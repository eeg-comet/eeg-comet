
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

        # Load model and do inference
        num_classes = self.best_maps.shape[0]
        maps = {i: chr(65 + i) for i in range(num_classes)}
        model_path = './models/model_v1.22.h5'
        model = load_model(model_path, compile=False)
        output = model.predict(image)
        label_result = self.get_labels(output, maps)

        micro_labels = []
        additional_label = chr(65 + 7)
        assert self.n_states < 27, 'Cannot label more than 27 microstates: not enough letters'

        for i in range(self.n_states):
            if i in label_result:
                micro_labels.append(label_result[i])
            else:
                micro_labels.append(additional_label)
                additional_label = chr(ord(additional_label) + 1)

        self.micro_labels = micro_labels
        print(f'Microstate Labels: {self.micro_labels}')

        ### TEMP
        # Save microstates as image
        # import os.path
        # import random
        # for step in range(15):
        #     for i in range(self.best_maps.shape[0]):
        #         filename_prefix = os.path.join(
        #             'C:/Users/amin_/OneDrive - Simon Fraser University (1sfu)/TOOLBOX/TRAIN_MICROSTATE_LABELER/MICROSTATES_AS_IMAGE/NEW/' + self.micro_labels[i], self.micro_labels[i])
        #         index = 400
        #         while True:
        #             filename = f"{filename_prefix}_{index}.png" if index > 1 else f"{filename_prefix}.png"
        #             if not os.path.exists(filename):
        #                 break
        #             index += 1
        #
        #         fig, ax = plt.subplots()
        #         random_contours = random.randint(0, 15)
        #         random_polarity = random.choice([1, -1])
        #         random_cmap = random.choice(['RdBu_r', 'coolwarm', 'bwr', 'seismic'])
        #         random_sensors = random.choice([True, False])
        #         random_interp = random.choice(['cubic', 'nearest', 'linear'])
        #         random_sphere = random.choice([None, 'auto', 'eeglab'])
        #         mne.viz.plot_topomap(random_polarity*self.best_maps[i, :], self.eeg_info,
        #                              contours=random_contours, sensors=random_sensors, axes=ax,
        #                              cmap=random_cmap, image_interp=random_interp, sphere=random_sphere, show=False)
        #         plt.savefig(filename, dpi=200, bbox_inches='tight')

        ### TEMP


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
                if state in maps:
                    result[image] = maps[state]
                else:
                    while True:
                        state = np.argsort(confidences[image])[::-1][0]  # Get next highest confidence
                        if state in maps:
                            result[image] = maps[state]
                            break
                        else:
                            confidences[image, state] = np.NINF
            else:
                confidences[image, state] = np.NINF
            if len(result) == confidences.shape[0] or len(result) == 7:
                break

        return result
