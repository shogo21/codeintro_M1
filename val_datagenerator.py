from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
import cv2
import random

class ValDataGenerator(Sequence):
    def __init__(self, data_list, labels, batch_size, image_size, max_sequence_num):
        self.data_list = data_list
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.data_list))
        self.max_sequence_num = max_sequence_num

    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data_list[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]

        images = []
        labels = []
        #bright = random.uniform(0.3, 2.0)
        for sequence in batch_data:
          sequence_images = [np.array(Image.open(img_path).resize(self.image_size)) for img_path in sequence]

          if len(sequence_images) < self.max_sequence_num:
            zero_imagearray = [np.zeros((64, 64, 3)) for i in range(self.max_sequence_num-len(sequence_images))]
            sequence_images = sequence_images + zero_imagearray

          #bright = random.uniform(0.3, 2.0)
          sequence_images = np.array(sequence_images)
          #sequence_images = cv2.convertScaleAbs(sequence_images, alpha=bright, beta=0)
          sequence_images = sequence_images / 255.0
          images.append(sequence_images)

        for label in batch_labels:
          if len(label) < self.max_sequence_num:
            label = label + [0]*(self.max_sequence_num-len(label))
          label = np.array(label)
          label = label.reshape((-1, 1))
          labels.append(label)
          """for i in range(1):
              labels.append(label)"""

        return np.array(images), np.array(labels)