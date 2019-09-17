import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, LinearSVC

THRESHOLD = 0.01


class Pipe:
    def __init__(self, reader):
        self.X_train, self.y_train = reader.get_train_data()

        self.cat_cols = [col for col in self.X_train.columns if self.X_train[col].dtype in ['object']]

        # self.num_cols = [col for col in self.X_train.columns if self.X_train[col].dtype in ['int64', 'float64']]
        self.num_cols = ['TSH', 'TT4', 'FTI']  # , 'T3']
        # top score: ['TSH', 'TT4', 'FTI'}

        # self.bol_cols = [col for col in self.X_train.columns if self.X_train[col].dtype in ['bool']]
        self.bol_cols = ['on thyroxine']  # , 'sick', 'psych']  # top score
        # top score: ['on thyroxine']

        self.cols = self.cat_cols + self.num_cols + self.bol_cols

    def get_pipe(self):
        num_transformer = Pipeline(steps=[
            ('imp', SimpleImputer(missing_values=np.nan, strategy='median', copy=False)),
            ('scaler', StandardScaler())
        ])

        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent', copy=False)),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, self.num_cols),
            ('cat', cat_transformer, self.cat_cols),
            ('var_threshold', VarianceThreshold(threshold=THRESHOLD), self.bol_cols),
        ])

        # feature selection zostal przeprowadzony w pliku testowym
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])

        return pipeline
