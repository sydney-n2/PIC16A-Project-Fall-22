import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# ignore some useless warnings which makes things ugly:
# import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None

class PokemonSet:
    """
        This class preprocess the Pokemon data, visualize the correlation between features, and train a Decision Tree to
        tell whether a Pokemon is legendary or not.
    Attributes:
        data (pd.DataFrame): the Pokemon data used
        feature (list): the features used for finding legendary Pokemons
    Methods:
        clean_data():
            Preprocess the data
        corr(**kwargs):
            Calculate and visualize correlation matrix
        split(**kwargs):
            Split dataset to train set and test set
        make_decision_tree_model(**kwargs):
            Generate and fit Decision Tree for the data
    """
    
    # default features used to train the model
    default_features = ["base_egg_steps","base_happiness","base_total","sp_attack"]
    
    def __init__(self, data, feature = default_features): 
        for f in feature:
            if type(f) not in [str, np.str_]:
                raise TypeError(f"Each feature should be a string, but a {type(f)} is passed.")
        feature = list(feature)
        feature.append('is_legendary')  # add the label (is_legendary) to the data, in addition to the features
        self.feature = feature
        try:
            self.data = data[feature]
        except:
            raise ValueError("Invalid feature is passed to the function")
            
    def clean_data(self):
        """
            Label-encode the string features
            Convert NaN or non-numeric values in numeric features to the mean value of that feature
        Returns:
            None
        """
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
    
    def corr(self, absolute=True, plot=True, values=False, clim=[0,1]):
        """
            Get the correlation matrix between different features
        Kwargs:
            absolute (bool): default=True. Returning the absolute-value correlation matrix
            plot (bool): default=True. Plot the matrix
            values (bool): default = False. Show values of correlation in the plot
            clim ([cMin, cMax]): default=[0,1]. The range of colorbar
        Returns:
            corr_matrix (pd.DataFrame): The correlation matrix
        """
        corr_matrix = self.data.corr(method='spearman')
        if absolute:
            corr_matrix = np.abs(corr_matrix)
        if plot:
            plt.figure(figsize=(10,10))
            plt.imshow(corr_matrix,cmap='Blues')
            plt.clim(clim)
            if values:
                for (i, j), z in np.ndenumerate(corr_matrix):
                    plt.text(j, i, '{:0.1f}%'.format(z*100), ha='center', va='center')
            else:
                plt.colorbar()
            plt.title("Correlation Matrix for Pokemon Dataset")
            plt.xticks(np.arange(len(self.feature)),self.feature,rotation=45)
            plt.yticks(np.arange(len(self.feature)),self.feature,rotation=45)
            plt.show()
        return corr_matrix
        
   
    def split(self, test_size=0.3, random_state=42):
        """
            Split the dataset into train and test sets.
        Kwargs:
            test_size (float): default = 0.3. the size of test set
            random_state (int): default = 42. the random seed for random splitting
        Returns:
            X_train(pd.DataFrame): the features for train set
            X_test(pd.DataFrame): the features for test set
            y_train(pd.Series): the labels for train set
            y_test(pd.Series): the labels for test set
        """
        X = self.data[self.feature[:-1]] # features
        y = self.data[self.feature[-1]]  # labels (is_legendary)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def make_decision_tree_model(self, depth=None, random_state = 42, plotting_enabled = True):
        """
            Create a model decision tree for dataset.
            If depth is unassigned, it will iterate depth from 1 to 20, plot the train and test scores,
            and use the depth with highest test score.
            The scores are calculated via 4-fold cross validation.
        Kwargs:
            depth (None or int): default = None
                          if None then select best depth from 1 to 20.
                          if int then pass it to the depth of decision tree.
            random_state (int):  default = 42
                          The random seed
        Returns:
            model_tree (tree.DecisionTreeClassifier):  the final tree model
        """
        
        def fit_tree(X, y, d):
            """
                Generate and fit a decision tree model for the data
            Args:
                X (pd.DataFrame): the features
                y (pd.DataFrame): the labels
                d (int): depth of tree
            Returns:
                model_tree (tree.DecisionTreeClassifier): the trained decision tree model
            """
            model_tree = tree.DecisionTreeClassifier(max_depth = d, random_state = random_state)
            model_tree.fit(X,y)
            return model_tree
        
        X_train, X_test, y_train, y_test = self.split()

        if depth is None:
            depths = range(1, 20)
            train_score = []
            test_score = []
            for d in depths:
                model_tree = fit_tree(X_train.values, y_train.values, d)
                train_score.append(cross_val_score(model_tree, X_train, y_train, cv=4).mean())
                test_score.append(cross_val_score(model_tree, X_test, y_test, cv=4).mean())
            if plotting_enabled is True:
                plt.scatter(depths, train_score,label="Train Score")
                plt.scatter(depths, test_score,label="Test Score")
                plt.title("Train and Test Score vs Tree Depth")
                plt.legend()
                plt.xlabel("Tree Depth")
                plt.ylabel("Score")
                plt.show()
            selected_depth = depths[np.argmax(test_score)]
            print(f"Using depth {selected_depth}")
        else:
            selected_depth = depth

        model_tree = fit_tree(X_train.values, y_train.values, selected_depth)

        # score model on testing and training data and print it out 
        print(f"Cross Validation Score on training data : {cross_val_score(model_tree, X_train, y_train, cv=4).mean()}")
        print(f"Cross Validation Score on testing data : {cross_val_score(model_tree, X_test, y_test, cv=4).mean()}")

        return model_tree
