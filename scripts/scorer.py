import os
import numpy as np
import pandas as pd

class Scorer():

    def __init__(self, fitted_model, x_test, y_test, class_encoding, save_path):
        self.model = fitted_model
        self.x_test = x_test
        self.y_test = y_test
        self.encoding = class_encoding
        self.save_path = save_path
    
    def get_accuracies(self, save = False, save_tags = {}):
        accuracies = {}
        accuracies['Full'] = self.model.score(
            self.x_test.values, 
            self.y_test)
        for class_id in np.unique(self.y_test):
            idxs = [True if v == class_id else False for v in self.y_test]
            features = self.x_test.values[idxs]
            labels = [class_id] * len(features)
            accuracies[self.encoding[class_id]] = self.model.score(features, labels)
        if save:
            save_tags.update(accuracies)
            df = pd.DataFrame(save_tags, index = [0])
            df.to_csv(self.save_path, mode = 'a', header = not os.path.exists(self.save_path))
        return accuracies

    