
# coding: utf-8

import sys
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred)
    m_recall = metrics.recall_score(actual, pred)
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)

def train_clf(train_data, train_tags):
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, np.asarray(train_tags))
    return clf


def main():
    filename = r"d:\tmp\book\ML\pytest\c5.txt"
    train_data = np.loadtxt(filename, delimiter = "," ,usecols=(2,4,14,8))
    print train_data

    train_tags = np.loadtxt(filename, delimiter = "," ,usecols=(6))

    print train_tags
    test_tags = [1, 2]
	
    test_data = [[ 13,70,1,68],
 [ 4,50,1,0]]
 
    clf = train_clf(train_data, train_tags)
    pred = clf.predict(test_data)
    print pred
    evaluate(np.asarray(test_tags), pred)



if __name__ == '__main__':
    main()
