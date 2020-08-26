import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


embed = pd.read_csv('graph_db1.csv',header=None,skiprows=[0])
price = pd.read_csv('item_price.csv')
mcat = pd.read_csv('mcat_graph.csv')
d = dict()
for i in range(len(mcat['name'])):
    d[mcat['name'].iloc[i]] = mcat['mcat'].iloc[i]

print(len(mcat['mcat'].unique()))
# print(price['price'].value_counts())
# print(type(price['price'][0]))
def build_dataset(df1,df2):
    #df1: embed df2:price
    df1.dropna(inplace = True)
    df2.dropna(inplace = True)
    merged = pd.merge(df1,df2,right_on='item_name',left_on=[0])
    #drop rows with item name as "no pc_item"
    merged.drop(merged[merged['item_name'] == 'no pc_item'].index, inplace=True)
    merged.drop(merged.columns[0], axis=1,inplace = True)
    # print(merged)
    return merged 

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,activation='sigmoid')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.01)

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['acc'])
  return model

data = build_dataset(embed,price)
l = []
for i in range(len(data['item_name'])):
    l.append(d[data['item_name'].iloc[i]])
data['target'] = l
print(data['target'].value_counts())
# data['target'] = (data.price>4000).astype(int)
# data.to_csv('lol.csv')

#test_train split
data.pop('item_name')
data.pop('price')
train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)
train_labels = train_dataset.pop('target')
test_labels = test_dataset.pop('target')

print('\nTrain data shape {}\nTrain label shape {}\nTest data shape {}\nTest label shape {}\n'.format(train_dataset.shape,train_labels.shape,test_dataset.shape,test_labels.shape))
print(train_dataset.head())
print(train_labels.head())

#Model
# model = build_model()
# model.summary()
# EPOCHS = 50
# history = model.fit(train_dataset, train_labels,epochs=EPOCHS, validation_split = 0.2, verbose=1)
# ypred = model.predict(test_dataset)
# ypred = np.where(ypred >= 0.5, 1, 0)
# print((ypred))
# from sklearn.metrics import accuracy_score,f1_score
# print(accuracy_score(test_labels,ypred))
# print(f1_score(test_labels,ypred,average='micro'))

#Model 2
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(verbose=2,n_estimators=80)
rf = rf.fit(train_dataset,train_labels)

ypred = rf.predict(test_dataset)

from sklearn.metrics import f1_score
print(rf.score(test_dataset,ypred))
print(f1_score(test_labels,ypred,average='micro'))
print(f1_score(test_labels,ypred,average='macro'))
print(f1_score(test_labels,ypred,average='weighted'))
