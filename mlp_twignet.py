ratio_of_train = 0.7
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
import warnings
from tensorflow import keras
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

# print(bert_embeddings)
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
    X.append(final_embeddings[i])
    y.append(my_labels[i])

df1 = pd.DataFrame(data, columns = ['embeddings', 'label'])

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=(1-ratio_of_train),shuffle = False)

print("Size of train =",len(X_train),"\t Size of test =",len(X_test))
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
print("Logistic regression Accuracy:", score)
print("========================")
print("==========SVM===========")
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print("Accuracy of SVM : ", accuracy(cm))
print("========================")


input_dim = len(X_train[1])
print("input dimension = ",input_dim)
model = Sequential()
opt = keras.optimizers.Adam(learning_rate=0.000001)
model.add(Dense(968, input_dim=input_dim, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
               optimizer=opt,
               metrics=['accuracy'])
model.summary()
history = model.fit(np.array(X_train), y_train,
                    epochs=3,
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

# print(input_dim)
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_dim=input_dim))
# model.add(Dropout(0.5))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()
# history = model.fit(np.array(X_train), y_train,
#                     epochs=3,
#                     verbose=1,
#                     validation_data=(np.array(X_test), y_test),
#                     batch_size=1)
# loss, accuracy = model.evaluate(np.array(X_train), y_train, verbose=1)
# print("========================")
# print("Training Accuracy 1d convolution: {:.4f}".format(accuracy))
# print("========================")
# loss, accuracy = model.evaluate(np.array(X_test), y_test, verbose=1)
# print("========================")
# print("Testing Accuracy 1d convolution:  {:.4f}".format(accuracy))
# print("========================")
