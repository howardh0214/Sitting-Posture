from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler

model = tf.keras.models.load_model('model_howard.h5')

model.summary()

raw_df = pd.read_csv("data/output_testing.csv")
new_df = pd.read_csv("data/output_testing.csv")

test_labels = np.array(raw_df.pop('yn'))
test_features = np.array(raw_df)

predictions = model.predict_classes(test_features)
predict = list(predictions)

expected = np.array(new_df.pop('yn'))
x = list(expected)

true_negative = 0
flase_positive = 0
false_negative = 0
true_positive = 0

for i in range(len(x)):
    print('Prediction is {} , expected {}'.format(predict[i], expected[i]))
    if int(predict[i]) == int(expected[i]) == 0:
        true_negative += 1
    elif int(predict[i]) == int(expected[i]) == 1:
        true_positive += 1
    elif int(predict[i]) != int(expected[i]) == 0:
        flase_positive += 1
    else:
        false_negative += 1

print("true_negative:", true_negative)
print("flase_positive:", flase_positive)
print("false_negative:", false_negative)
print("true_positive:", true_positive)

