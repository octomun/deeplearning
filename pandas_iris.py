import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/y/Desktop/080228-master/deeplearning/dataset/iris.csv',names=["sepal_length","sepal_width","petal_length","petal_width","species"])
sns.pairplot(df, hue='species')
#plt.show()

dataset=df.values
#데이터타입을 실수형으로
#x는 입력변수
#y는 결과변수
x = dataset[:,0:4].astype(float)
y_obj=dataset[:,4]
#-------------원 핫 인코딩 ----------------------
#문자열을 바꾸려면 클래스 이름도 숫자여야한다
#sklearn라이브러리의 LabelEncoder()함수를 사용
from sklearn.preprocessing import LabelEncoder

#array(['Iris-setosa'],['iris-versicolor''],['iris-virginica'])
#을 array([1,2,3])으로 변환
e= LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)
#활성화 함수로 사용할려면 y값이 0과 1로 이루어져야 한다
#array([1,2,3])을 array([[1,0,0],[0,1,0],[0,0,1]]로 변경

#from tensorflow.keras.utils import np_utils
y_encoded = tf.keras.utils.to_categorical(y)

#------------------------------------------------
#모델 생성 및 실행
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation='relu', input_dim=4))
# softmax는 입력값*가중치의 비율로  x1 + x2 + x3 = 1.0 이 되도록 하는 함수
# 큰값은 더 크게, 작은값은 더 작게 계산됨
# 반환값이 1, 0이 되는데 원핫인코딩한 [1,0,0]처럼 1, 0으로 분류해 품종을 예측
model.add(Dense(3, activation='softmax'))

#카테고리를 나눠야 하는 분류이므로 loss에 categorical을 쓴다
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#모델을 50회 반복 입력값은 1개
model.fit(x, y_encoded, epochs=50, batch_size=1)
print("\n accuracy : %.4f" % (model.evaluate(x, y_encoded)[1]))