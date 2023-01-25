from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pprint

from loader import Loader
from scorer import Scorer

# params
path = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/data'
loader = Loader(path)
pp = pprint.PrettyPrinter(indent = 4)

values_P = [2**n for n in range(1, 9)]
values_R = [2**n for n in range(1, 9)]
values_n_bins = [10*i for i in range(1, 21)]

for P in values_P:
    for R in values_R:
        for n_bins in values_n_bins:
            kwargs = {
                'P': P,
                'R': R,
                'method': 'uniform'
            }

            save_tags = {
                **kwargs,
                'n_bins': n_bins
            }

            # get train and test features
            features_train, names_train = loader.get_lbp_features(
                'train', 
                n_bins, 
                save = True, 
                save_tags = save_tags, 
                **kwargs)
            features_test, names_test = loader.get_lbp_features(
                'test', 
                n_bins,
                save = True, 
                save_tags = save_tags, 
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
                'feature_source': 'lbp', 
                'parameters': '_'.join([f'{name}:{values}' for name, values in save_tags.items()])}

            scorer = Scorer(model, x_test, y_test, loader.class_encoding, save_path)
            accuracies = scorer.get_accuracies(save, save_tags_score)

            print('\n')
            print('## Accuracies ##')
            pp.pprint(accuracies)




