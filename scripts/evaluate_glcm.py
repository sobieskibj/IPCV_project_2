from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pprint
import random

from loader import Loader
from scorer import Scorer

# params
path = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/data'
loader = Loader(path)
pp = pprint.PrettyPrinter(indent = 4)

values_distances = [[i for i in range(1, k + 1)] for k in range(1, 13)]
values_angles = [[i * np.pi/4 for i in range(k + 1)] for k in range(1, 13)]

random.shuffle(values_distances)
random.shuffle(values_angles)

for distances in values_distances:
    for angles in values_angles:
        kwargs = {
            'distances': distances,
            'angles': angles
        }

        feature_types = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

        # get train and test features
        features_train, names_train = loader.get_glcm_features(
            'train', 
            feature_types, 
            save = True, 
            save_tags = kwargs, 
            **kwargs)
        features_test, names_test = loader.get_glcm_features(
            'test', 
            feature_types, 
            save = True, 
            save_tags = kwargs, 
            **kwargs)

        # prepare data
        x_train = pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index')
        y_train = loader.get_encoded_classes('train')

        x_test = pd.DataFrame.from_dict(features_test, columns = names_train, orient = 'index')
        y_test = loader.get_encoded_classes('test')

        # fit model
        model = RandomForestClassifier(random_state = 0)
        model.fit(x_train.values, y_train)

        # calculate and save accuracies
        save_path = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/scores.csv'
        save = True
        save_tags_score = {
            'feature_source': 'glcm', 
            'parameters': '_'.join([f'{name}:{values}' for name, values in kwargs.items()])}

        scorer = Scorer(model, x_test, y_test, loader.class_encoding, save_path)
        accuracies = scorer.get_accuracies(save, save_tags_score)

        print('\n')
        print('## Accuracies ##')
        pp.pprint(accuracies)




