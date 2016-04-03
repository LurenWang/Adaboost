import scipy.io as scio
import numpy
import math
from decision_tree import BuildTree, N_ATTR
from decision_tree import Classify as StumpClassify
import pdb

def train_adaboost(trainData, numRounds):
    numSamples = trainData.shape[0]
    D = numpy.array([1/numSamples] * (trainData.shape[0]), dtype=numpy.float32)
    A = numpy.array([0] * numRounds, dtype=numpy.float32)
    F = [] #contains f_t which are the roots during each round
    _recurse_train(D, A, F, trainData, numRounds)
    return (F, A)

def _recurse_train(D, A, F, trainData, numRound):
    if (numRound == 0):
        return
    #multiply each column of D to each row of trainData except last column (classification)
    weightedData = numpy.multiply(trainData[:, :-1], D[:, numpy.newaxis])
    weightedData = numpy.column_stack((weightedData, trainData[:, -1]))
    root = BuildTree(weightedData, [], 1)
    F.append(root)
    z = 0
    classifications = numpy.array([0] * weightedData.shape[0])

    for i, row in enumerate(weightedData):
        classifications[i] = StumpClassify(row, root)
        #z_t = sum (D_t(x,y)*yf_t(x)
        z += D[i] * trainData[i][N_ATTR] * classifications[i]
    a = (0.5) * math.log((1 + z) / (1 - z))
    #not appending shitttt
    A[A.shape[0] - numRound] = a

    for i, weight in enumerate(D):
        #D_{t+1} = D_t * exp(-a_t * y * f_t(x))
        D[i] = D[i] * math.exp(-a * trainData[i][N_ATTR] * classifications[i])

    _recurse_train(D, A, F, trainData, numRound - 1)

def Classify(F, A, data, numRounds):
    sum = 0

    for i in range(numRounds):
        root = F[i]
        weight = A[i]
        classification = StumpClassify(data, root)
        sum += weight * classification
    
    return 1 if sum > 0 else 0

def ComputeError(F, A, numRounds, data):
    error = 0
    
    for i, row in enumerate(data):
        print(i)
        result = Classify(F, A, row, numRounds)
        if result != row[N_ATTR]:
            error += 1

    return error

if __name__=='__main__':
    trainData = scio.loadmat('spam.mat')['train_spam']
    testData = scio.loadmat('spam.mat')['test_spam']

    numRounds = 100
    F, A = train_adaboost(trainData, numRounds)
    #pdb.set_trace()
    error = ComputeError(F, A, numRounds, trainData)
    print(error)
