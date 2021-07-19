ratio_of_train = 5000/8000
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
from sklearn.ensemble import VotingClassifier
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow import keras
import os
import deslib
from deslib.des.knora_e import KNORAE
warnings.filterwarnings("ignore")
np.random.seed(7)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements


def ensemble(y_pred_MLP, y_pred_Logistic, y_pred_SVM):
    y_pred = []
    for i in range(len(y_pred_MLP)):
        count = int(y_pred_MLP[i]) + int(y_pred_SVM[i]) + int(y_pred_Logistic[i])
        if(count == 0 or count == 1):
            y_pred.append(0)
        else:
            y_pred.append(1)
    return y_pred

pickle_off = open ("Data/bert_large_embeddings_10000_experiment.txt", "rb")
bert_embeddings = pickle.load(pickle_off)

pickle_off1 = open ("Data/doc_vectors.txt", "rb")
gcn_embeddings = pickle.load(pickle_off1)


# pickle_off2 = open ("Data/label_dict.txt", "rb")
# dict = pickle.load(pickle_off2)
# print(dict)

shuffle = pd.read_csv("GCN/data/mr_shuffle.txt",sep='\t|\n',header=None,names=["row", "train_test", "label"])

# print(len(bert_embeddings))
# print(bert_embeddings)
my_dict = {}
my_labels = {}
tweet_label = []
for row_number, tweet in shuffle.iterrows():
    my_dict[tweet['row']] = gcn_embeddings[row_number]
    my_labels[tweet['row']] = tweet['label']
    tweet_label.append(tweet['label'])
final_embeddings = {}
final_embeddings_gcn = {}
final_embeddings_bert = {}
unique, frequency = np.unique(tweet_label[int(len(tweet_label)*ratio_of_train):], return_counts = True)
print(unique)
print(frequency)
print(len(shuffle))
for row_num in range(0,len(bert_embeddings)):
    # print(len(my_dict[row_num]),len(bert_embeddings[row_num]))
    a = []
    b = []
    c = []
    for i in my_dict[row_num] :
        a.append(i)
        c.append(i)
    # print(len(a))
    final_embeddings_gcn[row_num] = c
    for i in bert_embeddings[row_num]:
        a.append(i)
        b.append(i)
    final_embeddings_bert[row_num] = b
    # print(len(a))
    final_embeddings[row_num] = a
    # print(len(final_embeddings_gcn[row_num]))
# data = []
X=[]
X_gcn = []
X_bert = []
y=[]
for i in range(0,len(bert_embeddings)):
    # data.append([my_dict[i],my_labels[i]])
    X.append(final_embeddings[i])
    X_gcn.append(final_embeddings_gcn[i])
    X_bert.append(final_embeddings_bert[i])
    y.append(my_labels[i])

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# X = pca.fit_transform(X)


print(len(X[0]))


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=(1-ratio_of_train),shuffle = False)
X_train_gcn, X_test_gcn, y_train_gcn, y_test_gcn = train_test_split(X_gcn, y,random_state=1, test_size=(1-ratio_of_train),shuffle = False)
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(X_bert, y,random_state=1, test_size=(1-ratio_of_train),shuffle = False)


# print("Size of train =",len(X_train),"\t Size of test =",len(X_test))
# classifier = MLPClassifier(hidden_layer_sizes=(968,968), max_iter=200,activation = 'identity',solver='adam',random_state=1)
# classifier.fit(X_train, y_train)
# y_pred_MLP = classifier.predict(X_test)
#
# cm = confusion_matrix(y_test, y_pred_MLP)
# print("========================")
# print("Accuracy of MLPClassifier : ", accuracy(cm))
# print(classification_report(y_test, y_pred_MLP, digits=4))
# print("========================")
lb = LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
y_test = lb.transform(y_test)

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# clf = LinearDiscriminantAnalysis()
# clf.fit(X_train, y_train)
# y_pred_fld = clf.predict(X_test)
#


def svm_manual(X_train, X_test, y_train, y_test):
    lb = LabelBinarizer()
    lb.fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=200)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    from sklearn import svm
    # clf_SVM = svm.SVC(class_weight = 'balanced')
    clf_SVM = svm.SVC()
    clf_SVM.fit(X_train, y_train)
    y_pred_SVM = clf_SVM.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_SVM)
    print("Accuracy of SVM : ", accuracy(cm))
    print(classification_report(y_test, y_pred_SVM,  digits=4))
    print("========================")
    return y_pred_SVM, clf_SVM

ans, pred_1= svm_manual(X_train, X_test, y_train, y_test)
ans_gcn, pred_2 = svm_manual(X_train_gcn, X_test_gcn, y_train_gcn, y_test_gcn)
ans_bert, pred_3 = svm_manual(X_train_bert, X_test_bert, y_train_bert, y_test_bert)

from sklearn.decomposition import PCA
pca = PCA(n_components=200)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
model = KNORAE([pred_1, pred_2,pred_3], k = 3)
model.fit(X_train,y_train)
y_pred_knora = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_knora)
print("Accuracy of KNORAE : ", accuracy(cm))
print(classification_report(y_test, y_pred_knora,  digits=4))
print("========================")
counter = 0
gcn_count = 0
bert_count = 0

for i in range(len(ans_gcn)):

    if (ans_gcn[i] == y_test[i]) and (ans_bert[i]!= y_test[i]):
        counter+=1
        gcn_count+=1
        # print("gcn")
    elif((ans_gcn[i] != y_test[i]) and (ans_bert[i]== y_test[i])):
        counter+=1
        bert_count+=1
        # print('bert')

print("counter =", counter, "gcn count = ", gcn_count, "bert_count = ", bert_count)
# classifier_1 = LogisticRegression()
# classifier_1.fit(X_train, y_train)
# y_pred_Logistic = classifier_1.predict(X_test)
# # print(y_pred_Logistic)
# score = classifier_1.score(X_test, y_test)
# print("Logistic regression Accuracy:", score)
# print(classification_report(y_test, y_pred_Logistic,  digits=4))
# print("========================")
#
# # print(y_pred_SVM)
# y_pred_ens = ensemble(y_pred_MLP.tolist(), y_pred_Logistic.tolist(), y_pred_SVM.tolist())
# print("Accuracy of Ensemble : ", accuracy(confusion_matrix( y_test, y_pred_ens)))
# print(classification_report(y_test, y_pred_ens,  digits=4))
# print("========================")
# clf_RF = RandomForestClassifier(n_estimators=100)
# clf_RF.fit(X_train, y_train)
# y_pred_RF = clf_RF.predict(X_test)
# print("Accuracy of Random Forest : ", accuracy(confusion_matrix(y_test, y_pred_RF)))
# print(classification_report(y_test,y_pred_RF,  digits=4))
# print("========================")

#
# input_dim = len(X_train[1])
# print("input dimension = ",input_dim)
# model = Sequential()
# opt = keras.optimizers.Adam(learning_rate=0.000001)
# model.add(Dense(968, input_dim=input_dim, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#                optimizer=opt,
#                metrics=['accuracy'])
# model.summary()
# history = model.fit(np.array(X_train), y_train,
#                     epochs=3,
#                     verbose=1,
#                     validation_data=(np.array(X_test), y_test),
#                     batch_size=1)
# loss, accuracy = model.evaluate(np.array(X_train), y_train, verbose=1)
# print("========================")
# print("Training Accuracy: {:.4f}".format(accuracy))
# print("========================")
# loss, accuracy = model.evaluate(np.array(X_test), y_test, verbose=1)
# print("========================")
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# print("========================")
#
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
