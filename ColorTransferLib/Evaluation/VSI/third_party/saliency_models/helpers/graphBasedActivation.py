import numpy as np
import scipy.io
import sklearn.preprocessing
from . import markovChain
import os

def loadGraphDistanceMatrixFor28x32():
    current_file_path = __file__
    absolute_file_path = os.path.abspath(current_file_path)
    current_directory = os.path.dirname(absolute_file_path)
    parent_directory = os.path.dirname(current_directory)

    mat_path = os.path.join(parent_directory, "resources", "28__32__m__2.mat")

    #f = scipy.io.loadmat("ColorTransferLib/Evaluation/VSI/saliency_models/resources/28__32__m__2.mat")
    f = scipy.io.loadmat(mat_path)
    distanceMat = np.array(f['grframe'])[0][0][0]
    lx = np.array(f['grframe'])[0][0][1]
    dim = np.array(f['grframe'])[0][0][2]
    return [distanceMat, lx, dim]

def calculate(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major

    state_transition_matrix = Fab * np.abs(
        (np.zeros((distanceMat.shape[0], distanceMat.shape[1])) + map_linear).T - map_linear
    ).T

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain.solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped

def normalize(map, sigma):
    [distanceMat, _, _] = loadGraphDistanceMatrixFor28x32()
    denom = 2 * pow(sigma, 2)
    expr = -np.divide(distanceMat, denom)
    Fab = np.exp(expr)

    map_linear = np.ravel(map, order='F')  # column major
    # calculating STM : w = d*Fab
    state_transition_matrix = (Fab.T * np.abs(map_linear)).T

    # normalising outgoing weights of each node to sum to 1, using scikit normalize
    norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

    # caomputing equilibrium state of a markv chain is same as computing eigen vector of its weight matrix
    # https://lps.lexingtonma.org/cms/lib2/MA01001631/Centricity/Domain/955/EigenApplications%20to%20Markov%20Chains.pdf
    eVec = markovChain.solve(norm_STM, 0.0001)
    processed_reshaped = np.reshape(eVec, map.shape, order='F')

    return processed_reshaped
