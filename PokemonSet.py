import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
# ignore some useless warnings which makes things ugly:
# import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class PokemonSet(): 
    
    default_features = ["base_egg_steps","base_happiness","base_total","sp_attack"]
    
    def __init__(self, data, feature = default_features): 
        for f in feature:
            if type(f) not in [str]:
                raise TypeError(f"Each feature should be a string, but a {type(f)} is passed.")
        feature.append('is_legendary')  # add the label (is_legendary) to the data, in addition to the features
        self.feature = feature
        try:
            self.data = data[feature]
        except:
            raise ValueError("Invalid feature is passed to the function")
            
    def clean_data(self):
        '''
            Label-encode the string features
            Convert NaN or non-numeric values in numeric features to the mean value of that feature
        Args:
            None
        Returns:
            None
        '''
        str_feature = ['abilities', 'classfication', 'japanese_name', 'name', 'type1', 'type2']
        for sf in str_feature:
            if sf in self.feature:
                LE = LabelEncoder()
                self.data[sf] = LE.fit_transform(self.data[sf])
        for f in self.feature:
            # convert non-numeric values to NaN
            for i in range(len(self.data[f])):
                try:
                    self.data[f][i] = float(self.data[f][i])
                except:
                    self.data[f][i] = np.nan
            # convert all NaN to mean of that feature
            for i in range(len(self.data[f])):
                if np.isnan(self.data[f][i]):
                    self.data[f][i] = float(np.nanmean(self.data[f],dtype='float64'))
            # ensure dtype of DataFrame to be float
            self.data[f] = self.data[f].astype('float64')
    
    def corr(self, absolute=True, plot=True, clim=[0,1]): 
        '''
            Get the correlation matrix between different features
        Kwargs:
            absolute (bool): default=True. Returning the absolute-value correlation matrix
            plot (bool): default=True. Plot the matrix
            clim ([cMin, cMax]): default=[0,1]. The range of colorbar
        Returns:
            corr_matrix (pd.DataFrame): The correlation matrix
        ''' 
        corr_matrix = self.data.corr(method='spearman')
        if absolute:
            corr_matrix = np.abs(corr_matrix)
        if plot:
            plt.figure(figsize=(10,10))
            plt.imshow(corr_matrix,cmap='Blues')
            plt.title("Correlation Matrix for Pokemon Dataset")
            plt.xticks(np.arange(len(self.feature)),self.feature,rotation=45)
            plt.yticks(np.arange(len(self.feature)),self.feature,rotation=45)
            plt.clim(clim)
            plt.colorbar()
            plt.show()
        return corr_matrix
        
   
    def split(self, test_size=0.3, random_state=42): 
        '''
            Split the dataset into train and test sets.
        Kwargs:
            test_size (float): the size of test set
            random_state (int): the random seed for random splitting
        Returns:
            X_train(pd.DataFrame): the features for train set
            X_test(pd.DataFrame): the features for test set
            y_train(pd.Series): the labels for train set
            y_test(pd.Series): the labels for test set
        '''
        X = self.data[self.feature[:-1]] # features
        y = self.data[self.feature[-1]]  # labels (is_legendary)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    
    # put fit_tree here
    def fit_tree(self, X, y, d):
        '''
        insert a docstring here
        '''
        model_tree = tree.DecisionTreeClassifier(max_depth = d)
        model_tree.fit(X,y)
        return model_tree

    def make_decision_tree_model(self, depth=None):
        '''
            create a model decision tree for dataset
        Args:
            self:
        Returns:
            model_tree: a tree.DecisionTreeClassifier 
        '''
        X_train, X_test, y_train, y_test = self.split()

        if depth == None:
            depths = range(1, 20)
            max_test_score = -1
            best_depth = None
            for d in depths:
                model_tree = self.fit_tree(X_train, y_train, d)
                test_score = model_tree.score(X_test, y_test)
                if test_score > max_test_score:
                    max_test_score = test_score
                    best_depth = d
            selected_depth = best_depth
        else:
            selected_depth = depth

        #print(f"Using depth {selected_depth}")
        model_tree = self.fit_tree(X_train, y_train, selected_depth)

        # score model on testing data and print it out 
        #print(f"Score on testing data : {model_tree.score(X_test, y_test)}")

        return model_tree

def make_model_object(): 
    df = pd.read_csv("pokemon.csv") 
    ps = PokemonSet(data = df,feature = ["base_egg_steps","base_happiness","base_total","sp_attack","capture_rate"])
    ps.clean_data()
    X_train, X_test, y_train, y_test = ps.split()
    return ps.make_decision_tree_model()