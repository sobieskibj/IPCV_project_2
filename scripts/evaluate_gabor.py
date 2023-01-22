from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import pprint

from loader import Loader
from scorer import Scorer

path = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/data'
loader = Loader(path)
pp = pprint.PrettyPrinter(indent = 4)

kwargs = {
    'n_angles': 4, 
    'n_sigmas': 2,
    'frequencies': [0.05, 0.25]
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

encoded_classes_train = loader.get_encoded_classes('train')
encoded_classes_test = loader.get_encoded_classes('test')

x_train = pd.DataFrame.from_dict(features_train, columns = names_train, orient = 'index')
y_train = loader.get_encoded_classes('train')

x_test = pd.DataFrame.from_dict(features_test, columns = names_train, orient = 'index')
y_test = loader.get_encoded_classes('test')

model = RandomForestClassifier(random_state = 0)
model.fit(x_train.values, y_train)

scorer = Scorer(model, x_test, y_test, loader.class_encoding)
accuracies = scorer.get_accuracies()

print('\n')
print('## Accuracies ##')
pp.pprint(accuracies) # add tqdm




