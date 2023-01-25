import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

def process_glcm_str(data_glcm):
    all = {}
    for i, e in enumerate(data_glcm):
        splitted = [elem.split(':') for elem in e]
        values = {}
        for part in splitted:
            values[part[0]] = len(ast.literal_eval(part[1]))
        all[i] = values
    return all

def process_gabor_str(data_gabor):
    all = {}
    for i, e in enumerate(data_gabor):
        vals = [e[1], e[3], e[4]]
        splitted = [v.split(':') for v in vals]
        values = {}
        for part in splitted:
            values[part[0]] = len(ast.literal_eval(part[1])) if isinstance(ast.literal_eval(part[1]), list) else int(part[1])
        all[i] = values
    return all

def process_lbp_str(data_lbp):
    all = {}
    for i, e in enumerate(data_lbp):
        vals = [e[0], e[1], e[-1]]
        splitted = [v.split(':') for v in vals]
        values = {}
        for part in splitted:
            values[part[0]] = int(part[1])
        all[i] = values
    return all    

if __name__ == '__main__':

    plt.rcParams['font.size'] = '16'
    path_scores = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/scores/scores.csv'
    data = pd.read_csv(path_scores).drop(['Unnamed: 0'], axis = 1)
    group_names = {
        'lbp': 'LBP (459 samples)',
        'glcm': 'GLCM (97 samples)',
        'gabor': 'Gabor (18 samples)',
        'hist': 'Histogram-based (1 sample)'
    }

    data.feature_source = data.feature_source.replace(group_names)
    data = data[['feature_source', 'parameters', 'Full']]

    # glcm
    params_glcm = data[data.feature_source == 'GLCM (97 samples)'].parameters.apply(lambda x: x.split('_')).values

    glcm_param_vals = process_glcm_str(params_glcm)
    glcm_vals = pd.DataFrame.from_dict(glcm_param_vals, orient = 'index')
    glcm_vals['accuracy'] = data[data.feature_source == 'GLCM (97 samples)'].Full.values

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1, 2)
    sns.scatterplot(glcm_vals, x = 'distances', y = 'accuracy', ax = ax[0])
    ax[0].set_xlabel('Distances')
    ax[0].set_ylabel('Accuracy')
    sns.scatterplot(glcm_vals, x = 'angles', y = 'accuracy', ax = ax[1])
    ax[1].set_xlabel('Angles')
    ax[1].set_ylabel('')
    fig.suptitle("Accuracy in relation to different parameter values")

    # gabor
    params_gabor = data[data.feature_source == 'Gabor (18 samples)'].parameters.apply(lambda x: x.split('_')).values

    gabor_param_vals = process_gabor_str(params_gabor)
    gabor_vals = pd.DataFrame.from_dict(gabor_param_vals, orient = 'index')
    gabor_vals['accuracy'] = data[data.feature_source == 'Gabor (18 samples)'].Full.values

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1, 3)
    sns.scatterplot(gabor_vals, x = 'angles', y = 'accuracy', ax = ax[0])
    ax[0].set_xlabel('Angles')
    ax[0].set_ylabel('Accuracy')
    sns.scatterplot(gabor_vals, x = 'sigmas', y = 'accuracy', ax = ax[1])
    ax[1].set_xlabel('Sigmas')
    ax[1].set_ylabel('')
    sns.scatterplot(gabor_vals, x = 'frequencies', y = 'accuracy', ax = ax[2])
    ax[2].set_xlabel('Frequencies')
    ax[2].set_ylabel('')
    fig.suptitle("Accuracy in relation to different parameter values")

    # lbp
    params_lbp = data[data.feature_source == 'LBP (459 samples)'].parameters.apply(lambda x: x.split('_')).values

    lbp_param_vals = process_lbp_str(params_lbp)
    lbp_vals = pd.DataFrame.from_dict(lbp_param_vals, orient = 'index')
    lbp_vals['accuracy'] = data[data.feature_source == 'LBP (459 samples)'].Full.values

    sns.set_style('darkgrid')
    fig, ax = plt.subplots(1, 3)
    sns.scatterplot(lbp_vals, x = 'P', y = 'accuracy', ax = ax[0])
    ax[0].set_xlabel('P')
    ax[0].set_ylabel('Accuracy')
    sns.scatterplot(lbp_vals, x = 'R', y = 'accuracy', ax = ax[1])
    ax[1].set_xlabel('R')
    ax[1].set_ylabel('')
    sns.scatterplot(lbp_vals, x = 'bins', y = 'accuracy', ax = ax[2])
    ax[2].set_xlabel('Numer of bins')
    ax[2].set_ylabel('')
    fig.suptitle("Accuracy in relation to different parameter values")
    
    plt.show()
