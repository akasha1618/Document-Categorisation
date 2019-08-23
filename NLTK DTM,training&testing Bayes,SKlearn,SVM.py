######################################################################
#################Document Categorisation Two Classes
######################################################################
#Reading the files as corpus
import nltk
from nltk.corpus import PlaintextCorpusReader

doc_dirname_politics = "D:/Google Drive/Training/Datasets/News Group Data Text/mini_newsgroups/talk.politics.misc"
doc_dirname_comps = "D:/Google Drive/Training/Datasets/News Group Data Text/mini_newsgroups/comp.os.ms-windows.misc"

politics_news_corpus = PlaintextCorpusReader(doc_dirname_politics,'.*')
politics_news_corpus.fileids()
politics_news_corpus.raw('179097')
comp_news_corpus = PlaintextCorpusReader(doc_dirname_comps, '.*')

###############
##Preprocessing our corpus into documents
import re
from nltk.stem.porter import PorterStemmer
#from nltk.corpus import stopwords
#####Writing a Custom TOkenizer
stemmer = PorterStemmer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def custom_preprocessor(text):
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations
    text = nltk.word_tokenize(text)       #tokenizing
    text = [word for word in text if not word in stop_words] #English Stopwords
    text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
    return text
############
#testing our preprocessor
#custom_preprocessor(politics_news_corpus.raw('179097'))

politics_news_docs = [(custom_preprocessor(politics_news_corpus.raw(fileid)), 'politics')
                for fileid in politics_news_corpus.fileids()]
politics_news_docs[0]

comp_news_docs = [(custom_preprocessor(comp_news_corpus.raw(fileid)), 'comp')
                for fileid in comp_news_corpus.fileids()]
comp_news_docs[0]


############
#Merging our corpus into single documents object
documents = politics_news_docs + comp_news_docs
import random
random.seed(50)
random.shuffle(documents)

documents[0]
documents[1]
#################
#Building featuresets
#Building NLTK friendly Document Term Matrix(generally called featuresets)
all_words = []
for t in documents:
    for w in t[0]:
        all_words.append(w)


all_words_freq = nltk.FreqDist(all_words)
print(all_words_freq.most_common(150))
print(all_words_freq['president'])

#word_features = list(all_words_freq.keys())[:3000]
word_features = all_words
word_features
##defining a function to map our NLTK friendly dataset
def find_features(docs):
    words = set(docs)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#print((find_features(politics_news_corpus.words('176878'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
featuresets[1:5]

#################
#Creating training and testing data

train_set = featuresets[:160]
test_set = featuresets[160:]

#Building Naive Bayes Model
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('accuracy is : ', nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(25)


######################################################################
#################Text Categorisation Multi class Problem
######################################################################
import nltk
from nltk.corpus import PlaintextCorpusReader

doc_dirname_politics = "D:/Google Drive/Training/Datasets/News Group Data Text/mini_newsgroups/talk.politics.misc"
doc_dirname_comps = "D:/Google Drive/Training/Datasets/News Group Data Text/mini_newsgroups/comp.os.ms-windows.misc"
doc_dirname_rel = "D:/Google Drive/Training/Datasets/News Group Data Text/mini_newsgroups/talk.religion.misc"


politics_news_corpus = PlaintextCorpusReader(doc_dirname_politics,'.*')
comp_news_corpus = PlaintextCorpusReader(doc_dirname_comps, '.*')
religion_news_corpus = PlaintextCorpusReader(doc_dirname_rel, '.*')

###############
########

#custom_preprocessor(politics_news_corpus.raw('179097'))

politics_news_docs = [(custom_preprocessor(politics_news_corpus.raw(fileid)), 'politics')
                for fileid in politics_news_corpus.fileids()]

comp_news_docs = [(custom_preprocessor(comp_news_corpus.raw(fileid)), 'comp')
                for fileid in comp_news_corpus.fileids()]

religion_news_docs = [(custom_preprocessor(religion_news_corpus.raw(fileid)), 'religion')
                for fileid in religion_news_corpus.fileids()]

########
#Merging our corpus into single documents object
documents2 = politics_news_docs + comp_news_docs + religion_news_docs

#Shuffle documents.  
import random
random.seed(50)
random.shuffle(documents2)
#################
#Building NLTK friendly Document Term Matrix(generally called featuresets)
all_words2 = []
for t in documents2:
    for w in t[0]:
        all_words2.append(w)

#Word Freqeuncy
all_words2_freq = nltk.FreqDist(all_words2)
print(all_words2_freq.most_common(100))
print(all_words2_freq['comp'])

#word_features = list(all_words_freq.keys())[:3000]
word_features2 = all_words2
##defining a function to map our NLTK friendly dataset
def find_features2(docs):
    words = set(docs)
    features = {}
    for w in word_features2:
        features[w] = (w in words)
    return features

#print((find_features2(politics_news_corpus.words('176878'))))

featuresets2 = [(find_features2(rev), category) for (rev, category) in documents2]

#################
#Creating training and testing data

train_set2 = featuresets2[:240]
test_set2 = featuresets2[240:]


classifier = nltk.NaiveBayesClassifier.train(train_set2)
print('accuracy is : ', nltk.classify.accuracy(classifier, test_set2))
classifier.show_most_informative_features(25)


######################################################################
#################SKlearn Integraion for Binary classes
######################################################################
#################SVM for Text Categorisation 
######################################################################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

###### SVM Model with default parameters 
SVC_classifier = SklearnClassifier(SVC(kernel='rbf'))
SVC_classifier.train(train_set)
print("SVC_classifier accuracy on Train:", (nltk.classify.accuracy(SVC_classifier, train_set))*100)
print("SVC_classifier accuracy on Test:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)


###### SVM Model Fine tuning 
SVC_classifier = SklearnClassifier(SVC(kernel='rbf',gamma=0.01, C=0.6))
SVC_classifier.train(train_set)
print("SVC_classifier accuracy on Train:", (nltk.classify.accuracy(SVC_classifier, train_set))*100)
print("SVC_classifier accuracy on Test:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

###### SVM Model Fine tuning 
SVC_classifier = SklearnClassifier(SVC(kernel='rbf',gamma=0.001, C=0.6))
SVC_classifier.train(train_set)
print("SVC_classifier accuracy on Train:", (nltk.classify.accuracy(SVC_classifier, train_set))*100)
print("SVC_classifier accuracy on Test:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)


######################################################################
######RF Model Building
######################################################################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier

RF_classifier = SklearnClassifier(RandomForestClassifier(max_depth=1,n_estimators=20))
RF_classifier.train(train_set)
print("RF_classifier accuracy Train:", (nltk.classify.accuracy(RF_classifier, train_set))*100)
print("RF_classifier accuracy Test:", (nltk.classify.accuracy(RF_classifier, test_set))*100)

######################################################################
#################SKlearn Integraion for Multi classes
######################################################################
#################SVM for Text Categorisation 
######################################################################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

SVC_classifier = SklearnClassifier(SVC(kernel='rbf',gamma=0.006, C=0.5))
SVC_classifier.train(train_set2)
print("SVC_classifier accuracy Train:", (nltk.classify.accuracy(SVC_classifier, train_set2))*100)
print("SVC_classifier accuracy Test:", (nltk.classify.accuracy(SVC_classifier, test_set2))*100)

## SVM Fine-tuning
SVC_classifier = SklearnClassifier(SVC(kernel='rbf',gamma=0.004, C=0.6))
SVC_classifier.train(train_set2)
print("SVC_classifier accuracy Train:", (nltk.classify.accuracy(SVC_classifier, train_set2))*100)
print("SVC_classifier accuracy Test:", (nltk.classify.accuracy(SVC_classifier, test_set2))*100)



######################################################################
######RF Model Building
######################################################################
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier

RF_classifier = SklearnClassifier(RandomForestClassifier(max_depth=1,n_estimators=60))
RF_classifier.train(train_set2)
print("RF_classifier accuracy Train:", (nltk.classify.accuracy(RF_classifier, train_set2))*100)
print("RF_classifier accuracy Test:", (nltk.classify.accuracy(RF_classifier, test_set2))*100)
