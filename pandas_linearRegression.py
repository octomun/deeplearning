import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
df = pd.read_csv("C:/Users/y/Desktop/080228-master/deeplearning/dataset/housing.csv", delim_whitespace=True, header=None)

seed=0
np.random.seed(seed)
tf.random.set_seed(3)

dataset = df.values()
x = dataset[:, 0:13]
y = dataset[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, activation='relu', input_dim=13))
model.add(Dense(6, activation='relu'))
#선형회귀이기 때문에 마지막 출력층에 활성화 함수를 지정할 필요가 없다
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=200, batch_size=10)

#실제값과 예상가격을 비교하여 어느정도 학습됬는지 표시
y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = y_prediction[i]
    print("실제가격 : {:.3f}, 예상가격 : {:.3f}".format(label,prediction))

