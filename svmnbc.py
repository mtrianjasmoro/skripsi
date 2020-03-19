# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk

# Mendownload daftar kata yang ada (vocabulary)
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,precision_score
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
indo = stopwords.words('indonesian')

#preprosesing
dataset = pd.read_csv('training_gojek.csv')
corpus = []
for i in range(0, len(dataset)):
    #re = hapus / rubah
    review = re.sub('[^a-zA-Z]', ' ', dataset['komentar'][i])
    review = review.lower()
    review = review.split()    
    # Menghilangkan kata yang tidak ada di stopwords
#     psi = StemmerFactory()
#     ps = psi.create_stemmer()
#     review = [ps.stem(word) for word in review if not word in indo]
    # print(i)
    review = ' '.join(review)
    corpus.append(review)
class Analis:
    def __init__(self, training):
        self.training = training
            
        #tf-idf
        articles=np.array(corpus)
        labels=np.array(dataset['sentimen'])

        tf_vectorizer=TfidfVectorizer(min_df =4,max_df=0.3,ngram_range=(1,3))
        x_train_tfidf=tf_vectorizer.fit_transform(articles).toarray()
        x_test_tfidf=tf_vectorizer.transform(articles)

        X_train, X_test, y_train, y_test=train_test_split(x_train_tfidf,labels,test_size = training,random_state = 0)
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(x_train_tfidf,labels,test_size = training,random_state = 0)

        
    def svm(self,kernel):
        classifierSVM = SVC(kernel = kernel, random_state = 0, gamma='auto')
        classifierSVM.fit(self.X_train, self.y_train)
        y_pred_SVM = classifierSVM.predict(self.X_test)
        
        svm_hasil = []
        svm_hasil.append(accuracy_score(self.y_test, y_pred_SVM, normalize=True))
        svm_hasil.append(recall_score(self.y_test, y_pred_SVM, average='micro'))
        svm_hasil.append(precision_score(self.y_test, y_pred_SVM, average='micro'))
        
        return svm_hasil

    def nbc(self):
        classifierNB = GaussianNB()
        classifierNB.fit(self.X_train, self.y_train)
        y_pred_NB = classifierNB.predict(self.X_test)
        
        nbc_hasil = []
        nbc_hasil.append(accuracy_score(self.y_test, y_pred_NB, normalize=True))
        nbc_hasil.append(recall_score(self.y_test, y_pred_NB, average='micro'))
        nbc_hasil.append(precision_score(self.y_test, y_pred_NB, average='micro'))
        
        return nbc_hasil
    
