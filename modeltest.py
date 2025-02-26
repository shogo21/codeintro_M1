from tensorflow.keras.optimizers import Adamax, Adam
import tensorflow as tf
import prepare_data
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from vit_keras import vit, utils
from vit_datagenerator import CustomDataGenerator
import time
from scipy.stats import mode

WINDOW_SIZE = 29

def apply_mode_filter(data):
    half_window = int((WINDOW_SIZE-1)/2)
    
    for i in range(len(data)):
        if i > 0 and data[i] != data[i-1]:
            start = max(0, i-half_window)
            end = min(len(data), i+half_window+1)

            window_data = data[start:end]
            mode_value = mode(window_data)[0]
            data[i] = mode_value

    return data

loaded_model = tf.keras.models.load_model('./lstm_num14_yeschange/my_lstmmodel_num14_epo211_yesrandomchange')
loaded_model.compile(loss='mean_squared_error', optimizer = Adam(learning_rate=0.001), metrics=['accuracy'])

#testdata
DATA_DIR = './imagefile'
videos_test = sorted(glob.glob(DATA_DIR+'/taimatu/*')+glob.glob(DATA_DIR+'/sukegawa/*'))
inputs_test_path, labels_test_path = prepare_data.videotopath(videos_test)
inputs_test = prepare_data.imageread2(inputs_test_path)
print(len(inputs_test))


#padhing
sequence_X_test = prepare_data.padhing(inputs_test)
#standardization
sequence_X_test = prepare_data.normal(sequence_X_test)
#correctlabel
Y_test = prepare_data.correctlabel(labels_test_path)
#padhingu of correctlabel
sequence_Y_test = prepare_data.padhing(Y_test)




#test
score = loaded_model.evaluate(sequence_X_test, sequence_Y_test, batch_size=4)
print('loss: ', score[0])
print('accuracy: ', score[1])

#predict f1score

pred = loaded_model(sequence_X_test)
pred_touch = np.where(pred > 0.5, 1, 0)

start = time.time()

#モードフィルタ適用
chpred_touch = []
for data in pred_touch:
    chpred_touch.append(apply_mode_filter(data))
chpred_touch = np.array(chpred_touch) 

end = time.time()

print('time: ', end-start)

reshape_pred = pred_touch.reshape((sequence_X_test.shape[0], -1))
reshape_chpred = chpred_touch.reshape((sequence_X_test.shape[0], -1))

reshape_label = sequence_Y_test.reshape((sequence_X_test.shape[0], -1))
label_touch = np.where(reshape_label > 0.5, 1, 0)

join_label = []
join_predict = []
join_chpredict = []
for i in label_touch:
    join_label.extend(i)
for j in reshape_pred:
    join_predict.extend(j)
for k in reshape_chpred:
    join_chpredict.extend(k)

#正解ラベルと予測結果表示（1/0）
for i in range(sequence_X_test.shape[0]):
    print(join_label[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    #print(join_predict[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])
    print(join_chpredict[i*sequence_X_test.shape[1]:(i+1)*sequence_X_test.shape[1]])

#動画内で連続して1と予測されたフレーム数を出力
count_one = 0
touch = 'N'
for i, chp in enumerate(reshape_chpred):
    count_one = 0
    touch = 'N'
    for j in range(1, len(chp)):
        if ((chp[j]!=chp[j-1]) and touch == 'N'):
            touch = 'Y'
            count_one = count_one+1
        elif (touch == 'Y' and chp[j]==1):
            count_one = count_one+1
        elif (touch == 'Y' and chp[j]==0):
            print('i, count_one', i, count_one)
            count_one = 0
            touch = 'N'
        if ( j==(len(chp)-1) and count_one>0):
            print('i, count_one', i, count_one)
            count_one = 0
            touch = 'N'
#モードフィルタ適用前のF1スコアなど
join_cm = confusion_matrix(join_label, join_predict)
print(join_cm)
join_f1 = f1_score(join_label, join_predict)
print("f1:", join_f1)
join_precision = precision_score(join_label, join_predict)
print("precision: ", join_precision)
join_recall = recall_score(join_label, join_predict)
print("recall: ", join_recall)

#モードフィルタ適用後のF1スコアなど
chjoin_cm = confusion_matrix(join_label, join_chpredict)
print(chjoin_cm)
chjoin_f1 = f1_score(join_label, join_chpredict)
print("ch_f1:", chjoin_f1)
chjoin_precision = precision_score(join_label, join_chpredict)
print("ch_precision: ", chjoin_precision)
chjoin_recall = recall_score(join_label, join_chpredict)
print("ch_recall: ", chjoin_recall)


