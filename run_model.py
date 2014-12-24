import scipy.io as io
import numpy as np
import pickle
from helpers import *
from sklearn.externals import joblib

""" !!!IMPORTANT!!! 
	Before running this code,  please run the function ImageGaborize.m on the test data to gaborize.
    The test data will then be saved as "GaborizedImages.mat". The predictions, after running the model,
    will be stored in "test.csv" and "test.mat". 
"""

# Takes model and input and returns a 1 of K matrix of predictions given test data. 
def run_model(filename):
    test_data = io.loadmat(filename)
    test_images = test_data['GaborizedImages']
    model = joblib.load('trainedModel.pkl') 
    predictions = model.predict(test_images.T)
    filename = "test"
    # Create mat and csv to store the results
    create_csv(filename, predictions)
    create_mat(filename, predictions)
    return predictions.reshape(1, -1)

# Assuming the file name of the gaborized test images is "GaborizedImages.mat"
if __name__ == "__main__":
    # Example on running the model
    predictions = run_model("./GaborizedImages.mat")