import statistics
import numpy
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class evalMetrics:

    def __init__(self, labelNum):
        self.score = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.sumConMatrix = numpy.zeros([labelNum, labelNum])

    def addEval(self, score, labels_test, predLabels):
        precision = precision_score(labels_test, predLabels, average='weighted')*100
        recall = recall_score(labels_test, predLabels, average='weighted')*100
        f1 = f1_score(labels_test, predLabels, average='weighted')*100
        conMatrix = confusion_matrix(labels_test, predLabels)
        self.sumConMatrix = numpy.add(self.sumConMatrix, conMatrix)
        self.score.append(score*100)
        self.precision.append(precision)
        self.recall.append(recall)
        self.f1.append(f1)

    def getScore(self):
        return str(statistics.mean(self.score)), str(statistics.stdev(self.score))

    def getPrecision(self):
        return str(statistics.mean(self.precision)), str(statistics.stdev(self.precision))

    def getRecall(self):
        return str(statistics.mean(self.recall)), str(statistics.stdev(self.recall))

    def getF1(self):
        return str(statistics.mean(self.f1)), str(statistics.stdev(self.f1))

    def getConMatrix(self):
        return numpy.divide(self.sumConMatrix, 5)