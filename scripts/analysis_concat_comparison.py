import json
import pprint
import itertools
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from loader import Loader
from scorer import Scorer
from analysis_single_source_importance import *

path_data = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/data'
path_best_configs = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/best_configs.json'
with open(path_best_configs, 'r') as f:
    data = json.load(f)

loader = Loader(path_data)
pp = pprint.PrettyPrinter(indent = 4)
best_params_glcm = process_glcm_str([data['glcm'].split('_')], short = False)[0]
best_params_lbp = process_lbp_str([data['lbp'].split('_')])[0]
best_params_gabor = process_gabor_str([data['gabor'].split('_')], short = False)[0]

features_types = ['hist', 'glcm', 'gabor', 'lbp']
combs = []
for i in range(1, 5):
    combs += (list(itertools.combinations(features_types, i)))

results = {}

for combination in combs:
    x_train_full = []
    x_test_full = []

    if 'hist' in combination:
        print('Adding hist')

        features_train, names_train = loader.get_hist_features('train', save = True)
        features_test, names_test = loader.get_hist_features('test', save = True)

        x_train_full.append(pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index'))
        x_test_full.append(pd.DataFrame.from_dict(features_test, columns = names_test, orient = 'index'))

    if 'glcm' in combination:
        print('Adding glcm with params: ', best_params_glcm)
        
        feature_types = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

        features_train, names_train = loader.get_glcm_features(
            'train', 
            feature_types, 
            save = True, 
            save_tags = best_params_glcm, 
            **best_params_glcm)
        features_test, names_test = loader.get_glcm_features(
            'test', 
            feature_types, 
            save = True, 
            save_tags = best_params_glcm, 
            **best_params_glcm)

        x_train_full.append(pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index'))
        x_test_full.append(pd.DataFrame.from_dict(features_test, columns = names_test, orient = 'index'))

    if 'gabor' in combination:
        print('Adding gabor with params: ', best_params_gabor)

        kwargs = {
            'n_angles': best_params_gabor['angles'],
            'n_sigmas': best_params_gabor['sigmas'],
            'frequencies': best_params_gabor['frequencies']
        }

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

        x_train_full.append(pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index'))
        x_test_full.append(pd.DataFrame.from_dict(features_test, columns = names_train, orient = 'index'))

    if 'lbp' in combination:
        print('Adding lbp with params: ', best_params_lbp)

        kwargs = {
            'P': best_params_lbp['P'],
            'R': best_params_lbp['R'],
            'method': 'uniform'
        }

        save_tags = {
            **kwargs,
            'n_bins': best_params_lbp['bins']
        }

        features_train, names_train = loader.get_lbp_features(
            'train', 
            best_params_lbp['bins'], 
            save = True, 
            save_tags = save_tags, 
            **kwargs)
        features_test, names_test = loader.get_lbp_features(
            'test', 
            best_params_lbp['bins'],
            save = True, 
            save_tags = save_tags, 
            **kwargs)

        x_train_full.append(pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index'))
        x_test_full.append(pd.DataFrame.from_dict(features_test, columns = names_train, orient = 'index'))

    # get y
    y_train = loader.get_encoded_classes('train')
    y_test = loader.get_encoded_classes('test')

    # concat x's
    x_train_full = pd.concat(x_train_full, axis = 1)    
    x_test_full = pd.concat(x_test_full, axis = 1)

    # fit model
    model = RandomForestClassifier(random_state = 0)
    model.fit(x_train_full.values, y_train)

    # calculate accuracy
    scorer = Scorer(model, x_test_full, y_test, loader.class_encoding, save_path = '')
    accuracies = scorer.get_accuracies()

    print('## Accuracies ##')
    name = '+'.join(combination)
    print('\n')
    print('Combination: ', name)
    print('\n')
    pp.pprint(accuracies)
    results[name] = accuracies

results_all = pd.DataFrame.from_dict(results)
print(results_all)
