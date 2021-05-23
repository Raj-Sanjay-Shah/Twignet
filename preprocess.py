"""
This file is used to preprocess tweets in the following manner:
--Convert all tweets to string (even if they are numbers)
--Remove all duplicate letters in a any word: Words like 'Yessss' are converted to 'Yes'
--Replace all empty tweets with the string "Empty Tweet" (Any other Indentifier can be used)
--Remove all @USER and HTTPURL tags
--Convert all emojis to text in the following manner
--Decontraction of phrases
--Stop word removal
"""
#-----------------------------------------------------------------------------------------------
raw_input_file1 = "train.tsv"
raw_input_file2 = "valid.tsv"
raw_input_file3 = "test.tsv"
processed_input_file = "Data/preprocessed.tsv"
ratio_of_train = 0.7
#-----------------------------------------------------------------------------------------------
import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
import emoji
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.utils import shuffle
# df1 = pd.read_csv(raw_input_file1,sep='\t|\n',engine = 'python')
# df2 = pd.read_csv(raw_input_file2,sep='\t|\n',engine = 'python')
# df3 = pd.read_csv(raw_input_file3,sep='\t|\n',engine = 'python')
# tweets = pd.DataFrame(np.concatenate([ df2.values, df3.values]), columns=df2.columns)
tweets = pd.read_csv(raw_input_file3,sep='\t|\n',engine = 'python')
tweets['text'].fillna("Empty Tweet", inplace = True)
tweets['text'].apply(str)
from nltk.corpus import wordnet as wn
nltk.download('wordnet')

cachedStopWords = stopwords.words("english")

def rem_stop_words(text):
    text = ' '.join([word for word in text.split() if word not in cachedStopWords])
    return text

def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_str(string):
    string = string.strip().lower()
    string = rem_stop_words(string)
    string = decontracted(string)

    string = re.sub(r"<p>", " ", string)
    string = re.sub(r"</p>", " ", string)
    string = re.sub(r"\n", " ", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def withoutduplicates(string):
    prev=-1
    chars=[]
    for c in range(len(string)):
        if (c==0):
            prev = c
            chars.append(string[c])
        else:
            if (string[prev] == string[c]):
                continue
            else:
                chars.append(string[c])
                prev = c
    return ''.join(chars)

def preprocess_tweets(tweets):
    for row_number, tweet in tweets.iterrows():
        twitt = re.sub('@USER', '', tweet['text'])
        twitt = re.sub('HTTPURL', '', twitt)
        twitt = clean_str(twitt)
        words = twitt.split()

        for word in words:
            if isinstance(word, str) and word.isnumeric():
                word1 = withoutduplicates(word)
                if(word1 != word):
                    twitt = re.sub(re.escape(word), lambda _: word1,twitt)
        tweets.at[row_number,'text']= emoji.demojize(twitt)
tweets =shuffle(tweets)
preprocess_tweets(tweets)
tweets = tweets[:500]
print(tweets.head())
sentences =tweets['text'].tolist()
labels = tweets['Label'].tolist()
train_or_test_list = []
for i in range(0,int(ratio_of_train*len(labels))):
    train_or_test_list.append('train')
for i in range(int(ratio_of_train*len(labels)),len(labels)):
    train_or_test_list.append('test')

with open('Data/sentences.txt', 'wb') as fh:
   pickle.dump(sentences, fh)
with open('Data/train_or_test_list.txt', 'wb') as fh:
   pickle.dump(train_or_test_list, fh)
with open('Data/labels.txt', 'wb') as fh:
   pickle.dump(labels, fh)
tweets.to_csv(processed_input_file,sep = 'α',index= False)
