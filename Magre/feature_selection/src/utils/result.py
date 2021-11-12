import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR


class Result(object):
    """
    Common class to hold all the results of the synthetic experiments
    """
    def __init__(
            self, criterion, value, usefulFeatures=None,
            numUseful=None, binary=True, numNeighbors=-1, name=None):
        self.criterion = criterion
        self.value = value
        self.binary = binary
        self.numNeighbors = numNeighbors
        self.numUseful = numUseful
        self.name = name
        if usefulFeatures is not None:
            self.usefulFeatures = usefulFeatures
            self.numUsefulFeatures = len(usefulFeatures)

    def addSelectionResult(self, selected, Xtrain, Ytrain, Xtest, Ytest):
        self.selectCorrect = list(set(selected).intersection(self.usefulFeatures))
        self.selectedWrong = set(selected).difference(self.selectCorrect)
        self.numSelectedWrong = len(self.selectedWrong)
        self.computeAccuracy(selected, Xtrain, Ytrain, Xtest, Ytest)

    def computeAccuracy(self, selected, Xtrain, Ytrain, Xtest, Ytest):
        self.selected = np.array(list(selected))
        self.numSelected = len(self.selected)
        if self.binary:
            svmC1 = SVC(kernel="rbf", gamma='auto', C=1.0)
            svmHighC = SVC(kernel="rbf", gamma='auto', C=10.0)
            svmLowC = SVC(kernel="rbf", gamma='auto', C=0.1)
            # lm = LogisticRegression(solver='lbfgs')
        else:
            svmC1 = SVR(kernel="rbf", gamma='auto', C=1.0)
            svmHighC = SVR(kernel="rbf", gamma='auto', C=10.0)
            svmLowC = SVR(kernel="rbf", gamma='auto', C=0.1)
            # lm = LinearRegression()
        svmC1.fit(Xtrain[:, self.selected], Ytrain)
        svmHighC.fit(Xtrain[:, self.selected], Ytrain)
        svmLowC.fit(Xtrain[:, self.selected], Ytrain)
        # lm.fit(Xtrain[:, self.selected], Ytrain)
        self.svmC1Accuracy = svmC1.score(Xtest[:, self.selected], Ytest)
        self.svmHihgCAccuracy = svmHighC.score(Xtest[:, self.selected], Ytest)
        self.svmLowCAccuracy = svmLowC.score(Xtest[:, self.selected], Ytest)
        # self.lmAccuracy = lm.score(Xtest[:, self.selected], Ytest)
