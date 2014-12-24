import scipy.io
import numpy as np
from sklearn import cross_validation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from helpers import *

if __name__ == "__main__":
    # --------------- Data preparation --------------- #
    train = scipy.io.loadmat('./TrainGaborized.mat')
    public = scipy.io.loadmat('./PublicGaborized.mat')
    hidden = scipy.io.loadmat('./HiddenGaborized.mat')
    labeled_images = scipy.io.loadmat('./labeled_images.mat')
    train_labels = labeled_images['tr_labels']
    train_images = train['TrainImages']
    public_images = public['PublicImages']
    hidden_images = hidden['HiddenImages']
    # ------------------------------------------------ #
    # We performed a grid search to tune the hyperparameters for SVC
    svc = SVC(C=100, cache_size=500, class_weight='auto', coef0=0.0, degree=8, gamma=1.0000000000000001e-04,
              kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    engine = OneVsRestClassifier(svc)
    engine.fit(train_images.T, train_labels.reshape(-1))
    public_predictions = engine.predict(public_images.T)
    hidden_predictions = engine.predict(hidden_images.T)
    predictions = public_predictions
    for hidden_pred in hidden_predictions:
        predictions = np.append(predictions, hidden_pred)
    file_name = "svm_rbf_gabor"
    create_csv(file_name, predictions)
    create_mat(file_name, predictions)
    print "Starting cross validation..."
    kfold = cross_validation.KFold(train_labels.shape[0], n_folds=8, shuffle=True)
    scores = cross_validation.cross_val_score(engine, train_images.T, train_labels.reshape(-1), n_jobs=-1, cv=kfold)
    print 'Cross validation performances: ', scores
    print 'Average peroformance = ', sum(scores) / len(scores)
