import scipy.io
import numpy as np
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from helpers import *

if __name__ == "__main__":
    # --------------- Data preparation --------------- #
    labeled_data = scipy.io.loadmat('./labeled_images.mat')
    test_data = scipy.io.loadmat('./public_test_images.mat')
    test_images = test_data['public_test_images']
    train_labels = labeled_data['tr_labels']
    train_images = labeled_data['tr_images']
    identity = labeled_data['tr_identity']
    train_images = vectorize(normalize(train_images))
    test_images = vectorize(normalize(test_images))
    # ------------------------------------------------ #
    engine = OneVsRestClassifier(LinearSVC(C=0.3, loss='l2', tol=1e-15, random_state=0))
    engine.fit(train_images.T, train_labels.reshape(-1))
    test_predictions = engine.predict(test_images.T)
    file_name = "svm_linearsvc"
    create_csv(file_name, test_predictions)
    create_mat(file_name, test_predictions)
    print "Starting cross validation..."
    kf = cross_validation.KFold(train_labels.shape[0], n_folds=8, shuffle=True)
    scores = cross_validation.cross_val_score(engine, train_images.T, train_labels.reshape(-1), n_jobs=-1, cv=kf)
    print "Starting cross validation..."
    print 'Cross validation performances: ', scores
    print 'Average peroformance = ', sum(scores) / len(scores)
    

