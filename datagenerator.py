from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np
import cv2
import random

class CustomDataGenerator(Sequence):
    def __init__(self, data_list, labels, batch_size, image_size, max_sequence_num):
        self.data_list = data_list
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.indexes = np.arange(len(self.data_list))
        self.max_sequence_num = max_sequence_num
        self.on_epoch_end()  #first shuffle

    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))

    #バッチサイズごとにデータをロード
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = [self.data_list[i] for i in batch_indexes]
        batch_labels = self.labels[batch_indexes]

        images = []
        labels = []
        bright = random.uniform(0.3, 2.0)
        for sequence in batch_data:
          sequence_images = [np.array(Image.open(img_path).resize(self.image_size)) for img_path in sequence]

          #label = list(map(lambda p: (1 if "touch" in p else 0), sequence))

          if len(sequence_images) < self.max_sequence_num:
            zero_imagearray = [np.zeros((64, 64, 3)) for i in range(self.max_sequence_num-len(sequence_images))]
            sequence_images = sequence_images + zero_imagearray

          sequence_images = np.array(sequence_images)
          change_images1 = sequence_images / 255.0
          change_images1 = self.normal(change_images1)
          images.append(change_images1)

          #データ拡張
          change_images2 = cv2.convertScaleAbs(sequence_images, alpha=bright, beta=0)
          change_images2 = np.array(change_images2)
          change_images2 = change_images2 / 255.0
          change_images2 = self.normal(change_images2)
          images.append(change_images2)
            
        for label in batch_labels:
          if len(label) < self.max_sequence_num:
            label = label + [0]*(self.max_sequence_num-len(label))
          label = np.array(label)
          label = label.reshape((-1, 1))
          labels.append(label)
          for i in range(1):
              labels.append(label)

        return np.array(images), np.array(labels)

    #学習データシャッフル
    def on_epoch_end(self):
        #shuffle train data every one epoch
        np.random.shuffle(self.indexes)

    #標準化
    def normal(self, sequence):
        updated_sequence = []
        for i, data in enumerate(sequence):
            data = data.astype('float32')

            R,G,B = np.dsplit(data, 3)
            R = np.squeeze(R)
            if np.std(R) != 0:
                R = (R - np.mean(R)) / np.std(R)*0.166+0.5
            else:
                R = np.ones_like(R) * 0.5

            G = np.squeeze(G)
            if np.std(G) != 0:
                G = (G - np.mean(G)) / np.std(G)*0.166+0.5
            else:
                G = np.ones_like(G) * 0.5

            B = np.squeeze(B)
            if np.std(B) != 0:
                B = (B - np.mean(B)) / np.std(B)*0.166+0.5
            else:
                B = np.ones_like(B) * 0.5

            data = np.stack([R, G, B], 2)
            data = np.clip(data, 0.0, 1.0)
            
            updated_sequence.append(data)

        return np.array(updated_sequence)
