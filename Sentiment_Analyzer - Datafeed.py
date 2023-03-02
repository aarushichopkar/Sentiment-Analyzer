#Importing modules
import pandas as pd 
import numpy as np 
import re 
import matplotlib.pyplot as plt 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import csv

#Reading Data
df = pd.read_csv('dataset.csv')

#Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(lowercase=True, tokenizer = nltk.word_tokenize, stop_words= stopwords.words('english') , ngram_range=(1,2))

#Splitting Data into traing and testing
from sklearn.model_selection import train_test_split
x = df['Comment']
y_sentiment = df['Sentiment']
y_category = df['Category']
xtrain1, xtest1,ytrain1, ytest1 = train_test_split(x,y_sentiment, test_size=0.2, random_state=42, shuffle=True)

x_senti_vec = cv.fit_transform(xtrain1).toarray()
x_test_senti_vec = cv.transform(xtest1).toarray()

#Building a Multinomial Navies Baye's Classifier
from sklearn.naive_bayes import MultinomialNB
mn_senti = MultinomialNB()
mn_senti.fit(x_senti_vec,ytrain1)
y_pred_senti = mn_senti.predict(x_test_senti_vec)

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
accuracy_score(ytest1, y_pred_senti)
f1_score(ytest1, y_pred_senti, average=None)

#Splitting data 
xtrain2, xtest2, ytrain2, ytest2 = train_test_split(x,y_category, test_size=0.2, random_state=42, shuffle=True)

x_category_vec = cv.fit_transform(xtrain2).toarray()
x_test_category_vec = cv.transform(xtest2).toarray()

mn_category = MultinomialNB()
mn_category.fit(x_category_vec,ytrain2)
y_pred_category = mn_category.predict(x_test_category_vec)

from sklearn.metrics import f1_score, accuracy_score
accuracy_score(ytest2, y_pred_category)
f1_score(ytest1, y_pred_senti, average=None)

#Enter input data
inp = list(input("Enter the data(Comments) separated by Commas: \n").split(","))
inp_vec = cv.transform(inp).toarray()
s = mn_senti.predict(inp_vec)
c = mn_category.predict(inp_vec)

classification_result = []
for _ in range(len(inp)):
    print(inp[_] ,s[_] ,c[_])
    val = []
    val.append(inp[_])
    val.append(s[_])
    val.append(c[_])
    classification_result.append(val)

from tabulate import tabulate
print (tabulate(classification_result , headers=["Comment", "Sentiment", "Category"]))

with open('result.csv', 'w', newline='') as file:
     writer = csv.writer(file)
     writer.writerows(classification_result)