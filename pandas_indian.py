import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.random.set_seed(3)

dataset = numpy.loadtxt("C:/Users/y/Desktop/080228-master/deeplearning/dataset/pima-indians-diabetes.csv",delimiter=",")
x=dataset[:,0:8]
y=dataset[:,8]

model= Sequential()
model.add(Dense(12, input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=200, batch_size=10)

print("\n Accuracy : %.4f" % (model.evaluate(x,y)[1]))