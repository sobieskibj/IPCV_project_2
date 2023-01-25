from pathlib import Path
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import numpy as np
import pandas as pd
import cv2

class Loader():

    def __init__(self, path) -> None:
        self.path = path
        self.data_paths_dict = self.make_data_paths_dict()

    ## data paths dictionary ##
    def make_data_paths_dict(self):
        data_paths_dict = {} 
        for path_type in Path(self.path).iterdir():
            type = path_type.parts[-1]
            print('\n', f'Loading {type} set', '\n')
            data_paths_dict[type] = {}
            for i, path_class in enumerate(path_type.iterdir()):
                if path_class.parts[-1].startswith('KTH'):
                    class_name = path_class.parts[-1]
                    print(f'{i+1}. {class_name}')
                    data_paths_dict[type][class_name] = [*path_class.iterdir()]
                else:
                    print(f'{i+1}. Skipped due to wrong format')
                    continue
        return data_paths_dict

    ## sample image path ##
    @property
    def sample_image_path(self):
        for class_name in self.data_paths_dict['train']:
            for path in self.data_paths_dict['train'][class_name]:
                print('\n', f'Sample image path: {path}')
                return str(path)
    
    ## class encoding ##
    @property
    def class_encoding(self):
        return self._class_encoding
    
    @class_encoding.setter
    def class_encoding(self, encoding):
        self._class_encoding = encoding

    def get_encoded_classes(self, data_type):
        paths_per_class = self.data_paths_dict[data_type]
        encoded_classes = []
        if data_type == 'train':
            encoding = {}
            for i, (class_name, paths) in enumerate(paths_per_class.items()):
                encoding[i] = class_name
                encoded_classes += [i] * len(paths)
            self.class_encoding = encoding
        elif data_type == 'test':
            for i, (class_name, paths) in enumerate(paths_per_class.items()):
                encoded_classes += [i] * len(paths)
        return encoded_classes

    ## GLCM features ##
    # make
    def make_glcm_features(self, data_type, feature_types = ['contrast'], save = False, save_tags = {}, *args, **kwargs):
        paths_per_class = self.data_paths_dict[data_type]
        features = {}
        names = []
        init = True
        for paths in paths_per_class.values():
            for path in paths:
                img = cv2.imread(str(path))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                glcm_matrix = graycomatrix(img_gray, **kwargs)
                features[str(path)] = []
                for feature_type in feature_types:
                    feature_values = graycoprops(glcm_matrix, feature_type)
                    feature_values_fltn = feature_values.flatten()
                    features[str(path)] += list(feature_values_fltn)
                    if init: 
                        names += [f'{feature_type}_{k}'for k in range(len(feature_values_fltn))]
                init = False
        if save:
            df = pd.DataFrame.from_dict(features, columns = names, orient = 'index')
            path = self.get_features_path('glcm', data_type, save_tags)
            df.to_csv(path)
        return features, names

    # load or make if not already made
    def get_glcm_features(self, data_type, feature_types = ['contrast'], save = False, save_tags = {}, *args, **kwargs):
        path = self.get_features_path('glcm', data_type, save_tags)
        if path.exists():
            print('\n', 'GLCM features already created. Loading...')
            return self.load_features(path)
        else:
            print('\n', 'Creating GLCM features...')
            return self.make_glcm_features(data_type, feature_types, save, save_tags, *args, **kwargs)

    ## Gabor filters features
    # make
    def make_gabor_features(self, data_type, save = False, save_tags = {}, *args, **kwargs):
        paths_per_class = self.data_paths_dict[data_type]
        features = {}
        names = []
        init = True
        for paths in paths_per_class.values():
            for path in paths:
                img = cv2.imread(str(path))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                filters = self.make_filters(**kwargs)
                features[str(path)] = []
                for i, filter in enumerate(filters):
                    filtered = ndi.convolve(img_gray, filter, mode='wrap')
                    features[str(path)] += [filtered.mean(), filtered.var()]
                    if init: 
                        names += [f'gabor_mean_{i}', f'gabor_var_{i}']
                init = False
        if save:
            df = pd.DataFrame.from_dict(features, columns = names, orient = 'index')
            path = self.get_features_path('gabor', data_type, save_tags)
            df.to_csv(path)
        return features, names

    # load or make if not already made
    def get_gabor_features(self, data_type, save = False, save_tags = {}, *args, **kwargs):
        path = self.get_features_path('gabor', data_type, save_tags)
        if path.exists():
            print('\n', 'Gabor features already created. Loading...')
            return self.load_features(path)
        else:
            print('\n', 'Creating Gabor features...')
            return self.make_gabor_features(data_type, save, save_tags, *args, **kwargs)

    # make gabor filters
    def make_filters(self, n_angles, n_sigmas, frequencies):
        filters = []
        for theta in range(n_angles):
            theta = theta / n_angles * np.pi
            for sigma in (1, n_sigmas + 1):
                for frequency in frequencies:
                    filter = np.real(gabor_kernel(frequency, theta=theta,
                                                sigma_x=sigma, sigma_y=sigma))
                    filters.append(filter)
        return filters
        
    ## LBP features
    # make
    def make_lbp_features(self, data_type, n_bins, save = False, save_tags = {}, *args, **kwargs):
        paths_per_class = self.data_paths_dict[data_type]
        features = {}
        names = []
        init = True
        for paths in paths_per_class.values():
            for path in paths:
                img = cv2.imread(str(path))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                lbp = local_binary_pattern(img_gray, **kwargs)
                hist, _ = np.histogram(lbp, density = True, bins = n_bins, range = (0, n_bins))
                features[str(path)] = list(hist)
                if init: 
                    names += [f'lbp_hist_bin:{i}' for i in range(n_bins)]
                    init = False
        if save:
            df = pd.DataFrame.from_dict(features, columns = names, orient = 'index')
            path = self.get_features_path('lbp', data_type, save_tags)
            try:
                df.to_csv(path)
            except OSError:
                print('Filename too long, save failed.')
        return features, names

    # load or make if not already made
    def get_lbp_features(self, data_type, n_bins = 20, save = False, save_tags = {}, *args, **kwargs):
        path = self.get_features_path('lbp', data_type, save_tags)
        if path.exists():
            print('\n', 'LBP features already created. Loading...')
            return self.load_features(path)
        else:
            print('\n', 'Creating LBP features...')
            return self.make_lbp_features(data_type, n_bins, save, save_tags, *args, **kwargs)

    # Histogram-based features
    # make
    def make_hist_features(self, data_type, save = False, save_tags = {}, *args, **kwargs):
        paths_per_class = self.data_paths_dict[data_type]
        features = {}
        names = ['mean', 'variance', 'skewness', 'kurtosis', 'energy', 'entropy']
        for paths in paths_per_class.values():
            for path in paths:
                img = cv2.imread(str(path))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                n_bins = 256
                hist, _ = np.histogram(img_gray, density = True, bins = n_bins)
                bin_values = np.arange(0, 256, 1)
                mean = self.get_mean(hist, bin_values)
                variance = self.get_variance(hist, bin_values, mean)
                skewness = self.get_skewness(hist, bin_values, mean, variance)
                kurtosis = self.get_kurtosis(hist, bin_values, mean, variance)
                energy = self.get_energy(hist)
                entropy = self.get_entropy(hist)
                features[str(path)] = [mean, variance, skewness, kurtosis, energy, entropy]
        if save:
            df = pd.DataFrame.from_dict(features, columns = names, orient = 'index')
            path = self.get_features_path('hist', data_type, save_tags)
            try:
                df.to_csv(path)
            except OSError:
                print('Filename too long, save failed.')
        return features, names
    
    def get_hist_features(self, data_type, save = False, save_tags = {}, *args, **kwargs):
        path = self.get_features_path('hist', data_type, save_tags)
        if path.exists():
            print('\n', 'Histogram-based features already created. Loading...')
            return self.load_features(path)
        else:
            print('\n', 'Creating histogram-based features...')
            return self.make_hist_features(data_type, save, save_tags, *args, **kwargs)

    # load or make if not already made

    # histogram statistics utilities
    def get_mean(self, prob_density_estimate, bin_values):
        return np.average(prob_density_estimate, weights = bin_values)

    def get_variance(self, prob_density_estimate, bin_values, mean):
        weights = np.array([(e - mean)**2 for e in bin_values])
        return np.sum(prob_density_estimate * weights)
    
    def get_skewness(self, prob_density_estimate, bin_values, mean, variance):
        weights = np.array([(e - mean)**3 for e in bin_values])
        return variance ** (-3/2) * np.sum(weights * prob_density_estimate)
    
    def get_kurtosis(self, prob_density_estimate, bin_values, mean, variance):
        weights = np.array([(e - mean)**4 for e in bin_values])
        return variance ** (-3/2) * np.sum(weights * prob_density_estimate)

    def get_energy(self, prob_density_estimate):
        return np.sum(prob_density_estimate**2)

    def get_entropy(self, prob_density_estimate):
        return - np.sum(prob_density_estimate * np.log2(prob_density_estimate + 1))

    ## utilities ##
    def get_features_path(self, type, data_type, tags):
        tags = [f'{name}:{values}' for name, values in tags.items()] \
            if type != 'glcm' else [f'{name}:{len(values)}' for name, values in tags.items()]
        return Path(self.path).parent / 'features' / '_'.join([type, data_type] + tags)

    def load_features(self, path):
        dict = pd.read_csv(path, index_col = 0).to_dict('split')
        return {name: values for name, values in zip(dict['index'], dict['data'])}, dict['columns']


        
if __name__ == '__main__':
    path = '/Users/bartlomiejsobieski/Osobisty/VSC/Intro_to_img_processing/IPCV_project_2/data'
    loader = Loader(path)
    kwargs = {
        'distances': [1, 2],
        'angles': [0, np.pi/4]
    }
    feature_types = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    data_type = 'train'
    features, names = loader.get_glcm_features(data_type, feature_types, **kwargs)

    
    