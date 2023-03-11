import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
english_stopwords = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from nltk.tokenize import word_tokenize

train_data = pd.read_csv("nlp-getting-started/train.csv") # andmed mudeli treenimiseks
test_data = pd.read_csv("nlp-getting-started/test.csv") # andmed mudeli testimiseks
sample = pd.read_csv("nlp-getting-started/sample_submission.csv") # hiljem esitatav andmestik

train = train_data.drop(['location', 'keyword'], axis=1) # eemaldame tunnused, mida ei kasuta
# (võimalik, et mudel muutuks täpsemaks, kui kasutada mõnda välja jäetud tunnust ka, kuid selle täide viimisega ma kahjuks ei saanud hakkama)
test = test_data.drop(['location', 'keyword'], axis=1)

print(train['target'].value_counts(normalize=True)) # vaatame tunnuse 'target' jaotust


def puhastus(tweet): # andmete puhastamise (preprocessing) funktsioon
    tweet = tweet.lower() # läbivalt väiksed tähed
    lemmatizer = WordNetLemmatizer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet) # eemaldame lingid
    tweet = re.sub(r'#', '', tweet) #eemaldame trellid teemaviidete eest
    tweet = re.sub('[^A-Za-z]+', ' ', tweet) # eemaldame kõik, mis pole tähestiku tähed
    tokenized = word_tokenize(tweet) # sõna osadeks (tokens)
    tweets_clean = []
    for word in tokenized:
        if (word not in stopwords_english and  # eemaldame sõnad, mis pole mudeli koostamisel nii olulised, inglise keeles 'stopwords'
                word not in string.punctuation):  # eemaldame kirjavahemärgid
            stem_word = lemmatizer.lemmatize(word) # taastame sõna algkuju
            tweets_clean.append(stem_word)
    return " ".join(tweets_clean) # tagastame 'puhastatud' lause

train['puhastatud'] = train['text'].apply(lambda x: puhastus(x)) # rakendame eelnevat funkstiooni kõikidele ridadele andmestikus

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train['puhastatud']) # sõnaline tekst arvulisteks vektoriteks

X_train, X_test, y_train, y_test = train_test_split(train_x_vectors, train['target'], test_size=0.20, random_state=42)
# jaotame testimise andmestiku kaheks, et mudelit treenida ja testida

svm_mudel = svm.SVC() # kasutame SVM mudelit, sest andmestik ei ole väga suur ning see annab täpsemaid tulemusi kui mõni teine mudel, mida proovisin
svm_mudel.fit(train_x_vectors,train['target']) # treenime mudelit
vm_pred = svm_mudel.predict(X_test ) # ennustame mudeli põhjal

acc = accuracy_score(y_test,vm_pred) # antud juhul on 'accuracy score' mudeli hindamiseks sobilik, sest tunnusel
# 'target' on väärtuste osakaalud sarnased
print(f'Accuracy of model is',acc*100) # hinnang mudelile

y_pred_test = svm_mudel.predict(vectorizer.transform(test['text'].apply(lambda x: puhastus(x)))) # rakendame mudelit testimise andmestikul
submission = pd.DataFrame({'id': test['id'], 'target': y_pred_test}) # esitatav andmestik
submission.to_csv('submission.csv', index=False)

# esitamisel oli mudeli täpsuseks 0.8014
