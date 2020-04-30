# first neural network with keras tutorial
from numpy import loadtxt
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf

dataset = loadtxt('data.csv', delimiter=',')
# split into input (X) and output (y) variables
x = dataset[:, 0:32]
y = dataset[:, 32]

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)



# define the keras model
model = models.Sequential()
model.add(layers.Dense(64, input_dim=32, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=METRICS)
# fit the keras model on the dataset
model.fit(x, y, epochs=100, batch_size=32, shuffle=True)
# evaluate the keras model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))


model.save('model_pja.h5')
print("Saved model_pja.h5 to disk")