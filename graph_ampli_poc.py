# -*- coding: utf-8 -*-
"""Graph_ampli_POC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1l2o1xWy2QGUTCU9mdOWsWxKS1oDATTVQ
"""

# %tensorflow_version 1.x
!pip uninstall -y tensorflow
!pip install tensorflow-gpu==1.14
!pip install ampligraph

import sys,os
from google.colab import drive
drive.mount('/content/drive')

bpath = '/content/drive/My Drive/Stock Data' #datasetdir
sys.path.insert(0,bpath)
os.chdir(bpath)

import ampligraph
import pandas as pd
import numpy as np
import tensorflow as tf
ampligraph.__version__

x = pd.read_csv('triplet.csv')
x = x.values
x[:5, ]

entities = np.unique(np.concatenate([x[:, 0], x[:, 2]]))
entities

relations = np.unique(x[:, 1])
relations

from ampligraph.evaluation import train_test_split_no_unseen 

X_train_valid, X_test = train_test_split_no_unseen(x, test_size=3000)
X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=3000)
# X_train, X_test = train_test_split_no_unseen(x, test_size=3000)

print('Train set size: ', X_train.shape)
print('Test set size: ', X_test.shape)
print('Validation set size: ', X_valid.shape)

from ampligraph.latent_features import ComplEx

model = ComplEx(batches_count=100, 
                seed=0, 
                epochs=300, 
                k=150, 
                eta=5,
                optimizer='adam', 
                optimizer_params={'lr':1e-3},
                loss='multiclass_nll', 
                regularizer='LP', 
                regularizer_params={'p':3, 'lambda':1e-5}, 
                verbose=True)

positives_filter = x

# tf.logging.set_verbosity(tf.logging.ERROR)
# # model.fit(X_train, ,early_stopping = True)
# model.fit(X_train, early_stopping = True,early_stopping_params = \
#                   {
#                       'x_valid': X_valid,       # validation set
#                       'criteria':'hits1',         # Uses hits10 criteria for early stopping
#                       'burn_in': 100,              # early stopping kicks in after 100 epochs
#                       'check_interval':20,         # validates every 20th epoch
#                       'stop_interval':5,           # stops if 5 successive validation checks are bad.
#                       'x_filter': positives_filter,          # Use filter for filtering out positives 
#                       'corruption_entities':'all', # corrupt using all entities
#                       'corrupt_side':'s+o'         # corrupt subject and object (but not at once)
#                   }
#           )

from ampligraph.latent_features import save_model, restore_model
# save_model(model, './best_model.pkl')
model = restore_model('./best_model.pkl')
if model.is_fitted:
    print('The model is fit!')
else:
    print('The model is not fit! Did you skip a step?')

from ampligraph.evaluation import evaluate_performance
ranks = evaluate_performance(X_test, 
                             model=model, 
                             filter_triples=positives_filter,   # Corruption strategy filter defined above 
                             use_default_protocol=True, # corrupt subj and obj separately while evaluating
                             verbose=True)

from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

mrr = mrr_score(ranks)
print("MRR: %.2f" % (mrr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))

data = pd.read_csv('triplet.csv')
data.drop(data[data['name'] == 'no pc_item'].index, inplace=True)
data.drop(data[data['prop'] == 'no price'].index, inplace=True)
print(data.head())

import itertools
pcItem = data['name'].unique()
pcItem_embeddings = dict(zip(pcItem,model.get_embeddings(pcItem)))

ke = []
val = []
for k,v in pcItem_embeddings.items():
  ke.append(k)
  val.append(v)
embed_df = pd.DataFrame({'name':ke,'embed':val})


price_df = pd.read_csv('item_price.csv')
price_df.drop(price_df[price_df['item_name'] == 'no pc_item'].index, inplace=True)
price_df.drop(price_df[price_df['price'] == 'no price'].index, inplace=True)
price_df['embed'] = price_df['item_name'].apply(lambda x: pcItem_embeddings[x])

price_df1 = pd.DataFrame(price_df.embed.values.tolist()).add_prefix('embed_')
price_df1['price'] = price_df['price']
price_df = price_df1 
# print(price_df.loc[price_df['name'] == 'revolving computer chair'])

price_df.dropna(inplace = True)
price_df['target'] = (price_df.price>4000).astype(int)


print(price_df.head())
print(price_df.describe())
print(embed_df.head())
print(embed_df.describe())
# print(dict(itertools.islice(pcItem_embeddings.items(), 2)))

price_df['target'].value_counts()

train_dataset = price_df.sample(frac=0.8,random_state=0)
test_dataset = price_df.drop(train_dataset.index)
train_dataset.pop('price')
test_dataset.pop('price')

clf_X_train = train_dataset
clf_X_test = test_dataset
y_train = train_dataset.pop('target')
y_test = test_dataset.pop('target')

clf_X_train.shape, clf_X_test.shape

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(verbose=2,n_estimators=100)
rf = rf.fit(clf_X_train,y_train)

clf_model = XGBClassifier(n_estimators=500, max_depth=5, objective="binary:logistic",verbosity= 1)
clf_model.fit(clf_X_train, y_train)

from sklearn import metrics
# metrics.accuracy_score(y_test, clf_model.predict(clf_X_test))
print(metrics.accuracy_score(y_test, rf.predict(clf_X_test)))
print(metrics.accuracy_score(y_test, clf_model.predict(clf_X_test)))