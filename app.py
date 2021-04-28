import pandas as pd
data = pd.read_csv('C:/Users/y/Desktop/gpascore.csv')
print(data.isnull().sum())
#결측치 삭제
data = data.dropna()
#print(data.isnull().sum())
#결측치에 값 넣기
#data = data.fillna()

ydata = data['admit'].values
xdata = []
print(data)

#iterrow pandas데이터 프레임의 한 행씩 출력 결과
for i, rows in data.iterrows():
    xdata.append([rows['gre'],rows['gpa'],rows['rank']])

import numpy as np
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#모델 학습
model.fit(np.array(xdata),np.array(ydata),epochs=1000)

#예측
predict = model.predict([[750,3.70,3],[400,2.2,1]])
print(predict)