import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

path_scores = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/scores.csv'
data = pd.read_csv(path_scores).drop('Unnamed: 0', axis = 1)
best_configs = data[['feature_source', 'parameters', 'Full']].groupby('feature_source').\
    apply(lambda grp: grp.nlargest(1, columns = 'Full', keep = 'all'))[['feature_source', 'parameters']]
best_configs = {k: v for k, v in zip(best_configs.feature_source.values, best_configs.parameters.values)}

with open('/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/best_configs.json', 'w') as f:
    f.write(json.dumps(best_configs))