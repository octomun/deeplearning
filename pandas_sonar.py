import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

seed=0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("C:/Users/y/Desktop/080228-master/deeplearning/dataset/sonar.csv", header=None)
dataset = df.values
x = dataset[:,0:60]
y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)
#형식이 맞지 않아 float로 변환
x=np.asarray(x).astype(np.float32)
y=np.asarray(y).astype(np.float32)

'''
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(24, activation='relu', input_dim=60))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=100,batch_size=5)

#모델 저장
from keras.models import load_model
model.save('my_model.h5')

#삭제
del model

#불러오기
model= load_model('my_model.h5')

print("\n accuracy : %.4f" % (model.evaluate(xtest,ytest)[1]))
'''
# k겹 교차검증
from sklearn.model_selection import StratifiedKFold
n_fold = 10
skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

accuracy = []

for train, test in skf.split(x,y):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_dim=60))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    model.fit(x[train], y[train], epochs=100, batch_size=5)
    k_accuracy = "%.4f" % (model.evaluate(x[test],y[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy :" % n_fold, accuracy )
