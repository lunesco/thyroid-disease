import pandas as pd
import re

NEG_TARGET = 'negative'
HYPO_TARGET = 'hypothyroid'
NAMES_FILE_PATH = '..//Datasets//allhyper.names'


class DbReader:
    def __init__(self):
        self.allhyper_data_path = '..//Datasets//allhyper.data'
        self.allhyper_test_path = '..//Datasets//allhyper.test'
        self.allhypo_data_path = '..//Datasets//allhypo.data'
        self.allhypo_test_path = '..//Datasets//allhypo.test'

        # wczytujemy nazwy kolumn z pliku
        self.col_names = []
        with open(NAMES_FILE_PATH, 'r') as file:
            for line in file:
                regex = re.findall("^(.*):.*", line)
                if regex:
                    self.col_names.append(regex[0])
        self.col_names.append('diagnosis')

        allhyper_data = self.__read_csv(self.allhyper_data_path)
        allhyper_test = self.__read_csv(self.allhyper_test_path)
        allhypo_data = self.__read_csv(self.allhypo_data_path)
        allhypo_test = self.__read_csv(self.allhypo_test_path)

        self.__drop_columns(allhyper_data)
        self.__drop_columns(allhyper_test)
        self.__drop_columns(allhypo_data)
        self.__drop_columns(allhypo_test)

        hyper_replacement = {"T3 toxic": NEG_TARGET,
                             "goitre": NEG_TARGET,
                             "secondary toxic": NEG_TARGET}

        hypo_replacement = {"primary hypothyroid": HYPO_TARGET,
                            "compensated hypothyroid": HYPO_TARGET,
                            "secondary hypothyroid": HYPO_TARGET}

        allhyper_data.replace(to_replace=hyper_replacement, inplace=True)
        allhyper_test.replace(to_replace=hyper_replacement, inplace=True)
        allhypo_data.replace(to_replace=hypo_replacement, inplace=True)
        allhypo_test.replace(to_replace=hypo_replacement, inplace=True)

        self.X_train, self.y_train = self.__prepare_train_data(allhypo_data, allhyper_data)
        self.X_test, self.y_test = self.__prepare_test_data(allhypo_test, allhyper_test)

        self.cat_cols = [col for col in self.X_train.columns if self.X_train[col].dtype in ['object']]
        # self.__uncorrelate_data()  # nie wykrywa wtedy hyperthyroid

    def __read_csv(self, path):
        return pd.read_csv(path, sep=',|.\|', engine='python', names=self.col_names, index_col=(-1),
                           na_values='?', true_values=['t'], false_values=['f'])

    def __drop_columns(self, data, labels=None):
        columns = ['TSH measured', 'T3 measured', 'TT4 measured',
                   'T4U measured', 'FTI measured', 'TBG measured', 'TBG']
        if labels: columns = labels
        data.drop(columns=columns, inplace=True)

    def __prepare_train_data(self, hypo, hyper):
        y_hypo = hypo['diagnosis']
        y_hyper = hyper['diagnosis']
        X = hypo.drop(axis=1, columns='diagnosis')

        y = y_hyper.combine(y_hypo, lambda x, y: 'hyperthyroid' if 'hyper' in x else
        ('hypothyroid' if 'hypo' in y else 'negative'))

        self.X = X.copy()
        self.X['diagnosis'] = y

        return X, y

    def __prepare_test_data(self, hypo, hyper):
        y_hypo = hypo['diagnosis']
        y_hyper = hyper['diagnosis']
        X = hypo.drop(axis=1, columns='diagnosis')

        y = y_hyper.combine(y_hypo, lambda x, y: 'hyperthyroid' if 'hyper' in x else
        ('hypothyroid' if 'hypo' in y else 'negative'))
        X['diagnosis'] = y.copy()
        X.dropna(inplace=True)

        return X.drop(axis=1, columns='diagnosis'), X['diagnosis']

    def __uncorrelate_data(self):
        num_cols = ['int64', 'float64']
        numerical_cols = list(self.X_train.select_dtypes(include=num_cols).columns)
        num_X_train = self.X_train[numerical_cols]
        num_X_test = self.X_test[numerical_cols]

        correlated_features = set()
        correlation_matrix = num_X_train.corr()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.75:
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)

        self.X_train = pd.concat([num_X_train.drop(labels=correlated_features, axis=1),
                                  self.X_train.select_dtypes(exclude=num_cols)], axis=1)
        self.X_test = pd.concat([num_X_test.drop(labels=correlated_features, axis=1),
                                 self.X_test.select_dtypes(exclude=num_cols)], axis=1)

    def get_X(self):
        return self.X

    def get_all_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test
