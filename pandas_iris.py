import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
df=pd.read_csv('C:/Users/y/Desktop/080228-master/deeplearning/dataset/iris.csv',names=["sepal_length","sepal_width","petal_length","petal_width","species"])
sns.pairplot(df, hue='species')
#plt.show()

dataset=df.values
x = dataset[:,0:4].astype(float)
y_obj=dataset[:,4]

from sklearn.preprocessing import LabelEncoder

e= LabelEncoder()
e.fit(y_obj)
y = e.transform(y_obj)

from tensorflow.keras.utils import np_utils
y_encoded = tf.keras.utils.to_categorical(y)