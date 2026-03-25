import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
from utils import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        # Store embeddings
        self.embeddings = X
        
        self.targets = {}
        self.train_test_splits = {}
        
        for target_name, cols in Config.CHAINED_TARGETS.items():
            # Create concatenated target by joining non-null values
            y_values = df[cols].fillna('').apply(
                lambda row: '_'.join([str(val) for val in row if val != '']), 
                axis=1
            ).to_numpy()
            
            # Filter classes with at least 3 samples for stratified split
            y_series = pd.Series(y_values)
            good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
            
            if len(good_y_value) < 1:
                print(f"Target {target_name}: None of the classes have more than 3 records. Skipping...")
                self.train_test_splits[target_name] = None
                continue
            
            # Filter to keep only classes with sufficient samples
            y_good = y_values[y_series.isin(good_y_value)]
            X_good = X[y_series.isin(good_y_value)]
            
            # Calculate test size to maintain 20% of original data
            new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
            
            # Perform train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X_good, y_good, 
                test_size=new_test_size, 
                random_state=seed, 
                stratify=y_good
            )
            
            # Store splits for this target
            self.train_test_splits[target_name] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_all': y_good,
                'classes': good_y_value
            }
        
        # Set default target to y2 for backward compatibility
        if 'y2' in self.train_test_splits and self.train_test_splits['y2'] is not None:
            default_split = self.train_test_splits['y2']
            self.X_train = default_split['X_train']
            self.X_test = default_split['X_test']
            self.y_train = default_split['y_train']
            self.y_test = default_split['y_test']
            self.y = default_split['y_all']
        else:
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.y = None


    def get_type(self):
        return self.y
    
    def get_X_train(self, target='y2'):
        """Get X_train for specified target"""
        if target in self.train_test_splits and self.train_test_splits[target] is not None:
            return self.train_test_splits[target]['X_train']
        return self.X_train
    
    def get_X_test(self, target='y2'):
        """Get X_test for specified target"""
        if target in self.train_test_splits and self.train_test_splits[target] is not None:
            return self.train_test_splits[target]['X_test']
        return self.X_test
    
    def get_type_y_train(self, target='y2'):
        """Get y_train for specified target"""
        if target in self.train_test_splits and self.train_test_splits[target] is not None:
            return self.train_test_splits[target]['y_train']
        return self.y_train
    
    def get_type_y_test(self, target='y2'):
        """Get y_test for specified target"""
        if target in self.train_test_splits and self.train_test_splits[target] is not None:
            return self.train_test_splits[target]['y_test']
        return self.y_test
    
    def get_train_df(self):
        return self.train_df
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_type_test_df(self):
        return self.test_df
    
    def get_all_targets(self):
        """Get list of all available targets"""
        return [k for k, v in self.train_test_splits.items() if v is not None]
    
    def get_target_split(self, target):
        """Get complete split information for a specific target"""
        return self.train_test_splits.get(target)


