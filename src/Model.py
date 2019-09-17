from joblib import dump, load
from sklearn.svm import SVC


class SVMModel:
    def __init__(self):
        self.model = SVC(gamma='scale', probability=False)
        self.model_path = '..//Models//SVC'

    def set_params(self, params):
        self.model.set_params(**params)

    def get_model(self):
        return self.model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, y):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def save(self):
        dump(self.model, self.model_path)

    def load(self):
        self.model = load(self.model_path)
