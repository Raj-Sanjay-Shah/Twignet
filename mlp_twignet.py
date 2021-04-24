import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import warnings
import os
warnings.filterwarnings("ignore")
np.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

pickle_off = open ("Data/bert_embeddings.txt", "rb")
bert_embeddings = pickle.load(pickle_off)

pickle_off1 = open ("Data/doc_vectors.txt", "rb")
gcn_embeddings = pickle.load(pickle_off1)

shuffle = pd.read_csv("GCN/data/mr_shuffle.txt",sep='\t|\n',header=None,names=["row", "train_test", "label"])

my_dict = {}
my_labels = {}
for row_number, tweet in shuffle.iterrows():
    my_dict[tweet['row']] = gcn_embeddings[row_number]
    my_labels[tweet['row']] = tweet['label']
final_embeddings = {}

for row_num in range(0,len(bert_embeddings)):
#     print(len(my_dict[i]),len(bert_embeddings[i]))
    a = []
    for i in my_dict[row_num] :
        a.append(i)

    for i in bert_embeddings[row_num]:
        a.append(i)
    final_embeddings[row_num] = a
data = []
X=[]
y=[]
for i in range(0,len(bert_embeddings)):
    data.append([my_dict[i],my_labels[i]])
    X.append(my_dict[i])
    y.append(my_labels[i])

df1 = pd.DataFrame(data, columns = ['embeddings', 'label'])

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)

classifier = MLPClassifier(hidden_layer_sizes=(968,968), max_iter=200,activation = 'identity',solver='adam',random_state=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
print("========================")
print("Accuracy of MLPClassifier : ", accuracy(cm))
print("========================")
lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
print("========================")
print("Accuracy:", score)
print("========================")
input_dim = len(X_train[0])

model = Sequential()
model.add(Dense(10, input_dim=input_dim, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
model.summary()
history = model.fit(np.array(X_train), y_train,
                    epochs=20,
                    verbose=1,
                    validation_data=(np.array(X_test), y_test),
                    batch_size=1)
loss, accuracy = model.evaluate(np.array(X_train), y_train, verbose=1)
print("========================")
print("Training Accuracy: {:.4f}".format(accuracy))
print("========================")
loss, accuracy = model.evaluate(np.array(X_test), y_test, verbose=1)
print("========================")
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("========================")
