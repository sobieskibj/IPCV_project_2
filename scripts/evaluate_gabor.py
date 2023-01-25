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

values_n_angles = [i for i in range(1, 6)]
values_n_sigmas = [i for i in range(1, 6)]
values_frequencies = [[0.05, 0.25], [0.05, 0.25, 0.5], [0.05, 0.25, 0.5, 0.75]]

random.shuffle(values_n_angles)
random.shuffle(values_n_sigmas)
random.shuffle(values_frequencies)

for n_angles in values_n_angles:
    for n_sigmas in values_n_sigmas:
        for frequencies in values_frequencies:
            kwargs = {
                'n_angles': n_angles,
                'n_sigmas': n_sigmas,
                'frequencies': frequencies
            }

            # get train and test features
            features_train, names_train = loader.get_gabor_features(
                'train',
                save = True, 
                save_tags = kwargs, 
                **kwargs)
            features_test, names_test = loader.get_gabor_features(
                'test', 
                save = True, 
                save_tags = kwargs, 
                **kwargs)

            # preprare data
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
                'feature_source': 'gabor', 
                'parameters': '_'.join([f'{name}:{values}' for name, values in kwargs.items()])}

            scorer = Scorer(model, x_test, y_test, loader.class_encoding, save_path)
            accuracies = scorer.get_accuracies(save, save_tags_score)

            print('\n')
            print('## Accuracies ##')
            pp.pprint(accuracies)




