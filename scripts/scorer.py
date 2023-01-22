import numpy as np

class Scorer():

    def __init__(self, fitted_model, x_test, y_test, class_encoding):
        self.model = fitted_model
        self.x_test = x_test
        self.y_test = y_test
        self.encoding = class_encoding
    
    def get_accuracies(self):
        accuracies = {}
        accuracies['Full'] = self.model.score(
            self.x_test.values, 
            self.y_test)
        for class_id in np.unique(self.y_test):
            idxs = [True if v == class_id else False for v in self.y_test]
            features = self.x_test.values[idxs]
            labels = [class_id] * len(features)
            accuracies[self.encoding[class_id]] = self.model.score(features, labels)
        return accuracies

    