import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = '18'
path_scores = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/scores.csv'
data = pd.read_csv(path_scores).drop('Unnamed: 0', axis = 1)
group_names = {
    'lbp': 'LBP (459 samples)',
    'glcm': 'GLCM (97 samples)',
    'gabor': 'Gabor (18 samples)',
    'hist': 'Histogram-based (1 sample)'
}
data.feature_source = data.feature_source.replace(group_names)
sns.set_style("darkgrid")
fig = sns.catplot(
    data = data, y = "Full", x = 'feature_source', kind = 'box'
)
fig.set_axis_labels("Feature extraction method", 'Accuracy')
fig.fig.suptitle("Accuracy for each method with different parameter values")
fig.set_xticklabels(rotation = 30)
plt.show()