from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, Concatenate, GlobalAveragePooling2D, TimeDistributed, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from generator_convlstm import MultiSequenceDataGenerator
from datagenerator import CustomDataGenerator
from val_datagenerator import ValDataGenerator
import tensorflow as tf

import numpy as np
import cv2
import prepare_data
import random
import glob
import time

DATA_DIR = './imagefile'
videos = sorted(glob.glob(DATA_DIR+'/*/*'))


videos_test = []
for video in videos:
    if 'taimatu' in video or 'sukegawa' in video:
        videos_test.append(video)
for video_test in videos_test:
    videos.remove(video_test)



index = random.sample(range(len(videos)), k=14)
index = sorted(index, reverse=True)

videos_val = []
for i in index:
    videos_val.append(videos[i])
    del videos[i]


print(len(videos))
print(len(videos_test))
print(len(videos_val))

#videos = videos * 3
#videos_val = videos_val * 3
print(videos_val)
print(videos_test)
print(len(videos))
print(len(videos_test))
print(len(videos_val))



inputs_train_path, labels_train_path = prepare_data.videotopath(videos)

inputs_val_path, labels_val_path = prepare_data.videotopath(videos_val)

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 2
image_size = (64, 64)

max_sequence_num = 0
for data in inputs_train_path:
  max_sequence_num = max(max_sequence_num, len(data))

max_valsequence_num = 0
for data in inputs_val_path:
  max_valsequence_num = max(max_valsequence_num, len(data))

train_generator = CustomDataGenerator(inputs_train_path, labels_train_path, batch_size, image_size, max_sequence_num)
val_generator = ValDataGenerator(inputs_val_path, labels_val_path, batch_size, image_size, max_valsequence_num)
#for data in data_generator:
print(train_generator[0][0].shape)
print(val_generator[0][0].shape)


def inception_module(x, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters//4, (1, 1), padding='same', activation='relu')(x)

    # 3x3 Convolution
    conv3x3 = Conv2D(filters//4, (3, 3), padding='same', activation='relu')(x)

    # 5x5 Convolution
    conv5x5 = Conv2D(filters//4, (5, 5), padding='same', activation='relu')(x)

    # MaxPooling
    maxpool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    conv1x1_maxpool = Conv2D(filters//4, (1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate all branches
    output = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, conv1x1_maxpool])

    return output


def inception_encoder(image_size):
  input_layer = Input(shape=(image_size, image_size, 3))

  #conv1
  conv1 = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
  pool1 = MaxPooling2D((3, 3), strides=(2,2))(conv1)
  drop1 = Dropout(0.25)(pool1)

  # Inception Module
  inception_output1 = inception_module(drop1, filters=64)
  inception_output2 = inception_module(inception_output1, filters=128)

  pool2 = MaxPooling2D((3, 3), strides=(2,2))(inception_output2)
  drop2 = Dropout(0.25)(pool2)

  # Inception Module
  inception_output3 = inception_module(drop2, filters=256)
  inception_output4 = inception_module(inception_output3, filters=512)

  #GlobalAveragePooling
  output = GlobalAveragePooling2D()(inception_output4)
  drop3 = Dropout(0.5)(output)

  # Create the model
  model = Model(inputs=input_layer, outputs=drop3)

  #model.summary()

  return model

cnn_input = Input(shape=(None, 64, 64, 3), name='cnn_input')
cnn_model = inception_encoder(64)


cnn_output = TimeDistributed(cnn_model, name='time_distributed')(cnn_input)
lstm_output = LSTM(512, stateful=False, return_sequences=True, name='lstm')(cnn_output)
dropout_lstm_inception = Dropout(0.5, name='dropout')(lstm_output)
dense_lstm = Dense(1, activation='sigmoid', name='dense')(dropout_lstm_inception)

final_model = Model(inputs=cnn_input, outputs=dense_lstm)

final_model.summary()

#loaded_model = tf.keras.models.load_model('./lstm_num14_yeschange/my_lstmmodel_num14_epo211_yesrandomchange_statefultrue')
#loaded_model.summary()

epochs = 200

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1, mode='max', min_lr=0.0001)

checkpoint = ModelCheckpoint(filepath='./lstm_num14_yeschange/my_lstmmodel_num14_epo{epoch:03d}_yesrandomchange', save_weights_only=False, save_best_only=True, monitor='val_loss', verbose=1)

#early = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto', restore_best_weights=True)

tensorboard = TensorBoard(log_dir='./lstm_num14_yeschange/my_lstmmodel_num14_epo_yesrandomchange_logs', histogram_freq=1)
callbacks_list=[reduce_lr, checkpoint, tensorboard]
#callbacks_list=[early, checkpoint, tensorboard]


final_model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.001), metrics=['accuracy'])
#loaded_model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.0001), metrics=['accuracy'])

start = time.time()
#with tf.device('/GPU:0'):
history = final_model.fit(train_generator, epochs=800, batch_size=1, verbose=1, validation_data=val_generator, callbacks=callbacks_list)
#history = loaded_model.fit(train_generator, epochs=800, batch_size=1, verbose=1, validation_data=val_generator, callbacks=callbacks_list)


end = time.time()

print('時間: ', end-start)

# モデルと学習履歴を保存　saveでも保存可能,モデルの重みも保存される、損失関数、最適化関数も保存される
final_model.save("./lstm_num14_yeschange/my_lstmmodel_num14_epo800_yesrandomchange")
#loaded_model.save("./lstm_num14_yeschange/my_lstmmodel_num14_epo800_yesrandomchange")
#学習履歴を保存する、つまりhistoryを保存する
with open('./lstm_num14_yeschange/my_lstmmodel_num14_epo800_yesrandomchange.pkl', 'wb') as history_file:
    import pickle
    pickle.dump(history.history, history_file)
