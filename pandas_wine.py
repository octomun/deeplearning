from keras.models import Sequential
from keras.layers import Dense
#ModelCheckpoint 모델 저장시 epoch마다 정확도를 함께 기록하면서 저장
#EarlyStopping 테스트 오차가 줄지 않으면 테스트 종료 함수
from keras.callbacks import ModelCheckpoint, EarlyStopping

import os
import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)

# 데이터 입력
df_pre = pd.read_csv('C:/Users/y/Desktop/080228-master/deeplearning/dataset/wine.csv', header=None)
#frac = 1 원본데이터의 100%를 불러오는
df = df_pre.sample(frac=0.15)

dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

# 모델 설정
model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#모델 컴파일
model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

#학습 자동중단 설정 ( 오차율이 줄어들지 않으면)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)

#모델 저장폴더 설정
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

#모델 저장 조건
#케라스 내부에서 각종 변수가 있다
# - loss : 훈련 손실값
# - acc : 훈련 정확도
# - val_loss : 검증 손실값
# - val_acc : 검증 정확도
modelpath='./model/{epoch:02d}-{val_loss:.4f}.hdf5'
#verbose가 1이면 함수의 진행사항 출력, 0이면 생략
#save_best_only로 모델이 나아졌을 때를 대상으로 함
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 모델 실행 및 저장
#model.fit(X, Y, validation_split=0.2, epochs=2000, batch_size=200, callbacks=[early_stopping_callback])
#model.fit(X, Y, validation_split=0.2, epochs=200, batch_size=200, verbose=0, callbacks=[checkpointer])
model.fit(X, Y, validation_split=0.2, epochs=3500, batch_size=500, verbose=0, callbacks=[early_stopping_callback,checkpointer])
'''
history = model.fit(X, Y, validation_split=0.33, epochs=3500, batch_size=500)
#오차값
y_vloss=history.history['val_loss']
#정확도
y_acc=history.history['accuracy']

x_len = numpy.arange(len(y_acc))
#c : color
plt.plot(x_len,y_vloss,'o',c ='red', markersize = 3)
plt.plot(x_len,y_acc,'o',c ='blue', markersize = 3)
plt.show()
'''
print("\n accuracy : %.4f" % (model.evaluate(X,Y)[1]))