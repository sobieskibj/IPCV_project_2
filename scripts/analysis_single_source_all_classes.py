import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '16'
path_scores = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/scores.csv'
data = pd.read_csv(path_scores).drop(['Unnamed: 0', 'parameters', 'Full'], axis = 1)
group_names = {
    'lbp': 'LBP (459 samples)',
    'glcm': 'GLCM (97 samples)',
    'gabor': 'Gabor (18 samples)',
    'hist': 'Histogram-based (1 sample)'
}
data.feature_source = data.feature_source.replace(group_names)
data_transformed = pd.melt(data, id_vars = ['feature_source'], value_vars = data.columns.drop('feature_source').to_list())
sns.set_style("darkgrid")
hue_order = data_transformed[data_transformed.feature_source == 'Histogram-based (1 sample)'].sort_values('value').variable.to_list()
fig = sns.catplot(
    data = data_transformed, y = 'value', x = 'feature_source', kind = 'box', hue = 'variable', hue_order = hue_order
)
fig.set_axis_labels('Feature extraction method', 'Accuracy')
fig.fig.suptitle("Accuracy for each method with different parameter values by class")
fig.set_xticklabels(rotation = 30)
plt.show()