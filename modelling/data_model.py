import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Store embeddings
        self.embeddings = X

        df_complete = df.dropna(subset=['y2', 'y3', 'y4']).reset_index(drop=True)
        X_complete = X[df_complete.index] if hasattr(X, '__len__') else X

        # Re-align X to the filtered df rows using original positions
        original_indices = df_complete.index.tolist()
        X_complete = X[original_indices]

        print(f"[Data] {len(df)} total rows → {len(df_complete)} rows with complete y2+y3+y4 labels")

        y2 = df_complete['y2'].to_numpy()
        y2_series = pd.Series(y2)
        good_y2 = y2_series.value_counts()[y2_series.value_counts() >= 3].index

        if len(good_y2) < 1:
            print("None of the classes have more than 3 records. Skipping...")
            self.X_train = None
            self.X_test  = None
            self.y_train = None
            self.y_test  = None
            self.y = None
            self.train_test_splits = {}
            return

        mask = y2_series.isin(good_y2)
        X_good  = X_complete[mask.values]
        df_good = df_complete[mask.values].reset_index(drop=True)
        y2_good = df_good['y2'].to_numpy()

        new_test_size = len(df_complete) * 0.2 / len(df_good)

        (self.X_train, self.X_test,
         train_df, test_df) = train_test_split(
            X_good, df_good,
            test_size=new_test_size,
            random_state=seed,
            stratify=y2_good
        )

        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        self.y   = y2_good

        self.train_test_splits = {}

        for target_name, cols in Config.CHAINED_TARGETS.items():
            y_train = train_df[cols].apply(
                lambda row: '_'.join([str(v) for v in row if str(v) != '']), axis=1
            ).to_numpy()

            y_test = test_df[cols].apply(
                lambda row: '_'.join([str(v) for v in row if str(v) != '']), axis=1
            ).to_numpy()

            self.train_test_splits[target_name] = {
                'X_train': self.X_train,
                'X_test':  self.X_test,
                'y_train': y_train,
                'y_test':  y_test,
                'classes': pd.Series(y_train).value_counts().index
            }

        self.y_train = self.train_test_splits['y2']['y_train']
        self.y_test  = self.train_test_splits['y2']['y_test']

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_embeddings(self):
        return self.embeddings

    def get_all_targets(self):
        """Return list of target names that have valid splits."""
        return [k for k, v in self.train_test_splits.items() if v is not None]

    def get_target_split(self, target):
        """Return the full split dict for a specific target."""
        return self.train_test_splits.get(target)