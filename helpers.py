import pickle
import numpy as np
import scipy.io as io
from sklearn.externals import joblib

# Save training model 
def save_model(model):
    joblib.dump(model, 'trainedModel.pkl') 
    print "Successfully saved model"

# Reshape a image matrix into a vector 
def vectorize(images):
    temp_images = np.zeros((np.shape(images)[0] * np.shape(images)[1], np.shape(images)[2]))
    for i in xrange(np.shape(images)[2]):
        temp_images[:, i] = images[:, :, i].reshape(-1, 1)[:, 0]
    return temp_images

# Normalize input images
def normalize(images):
    images = images[:, :, :].astype('float64')
    images = images / 255
    return images

# Create a mat file for our predictions
def create_mat(filename, predictions):
    results = np.array(predictions.reshape(1, -1)[0])
    temp = {}
    temp[filename] = results
    io.savemat(filename, temp)
    filename = filename + ".mat"
    print "Created " + filename

# Create csv file for our predictions 
def create_csv(filename, predictions):
    filename = filename + ".csv"
    predictions = predictions.reshape(-1)
    if (len(predictions) < 1253):
        zeros = np.zeros(1253 - len(predictions))
        predictions = np.append(predictions, zeros).reshape(-1)

    f = open(filename, 'w')
    f.write('Id,Prediction\n')
    for i in range(0, len(predictions)):
        s = '{0},{1}\n'.format(i + 1, int(predictions[i]))
        f.write(s)
    print "Created " + filename 