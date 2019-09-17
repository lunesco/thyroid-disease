from time import time
import warnings
from joblib import dump
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.DbReader import DbReader
from src.Model import SVMModel
from src.Preprocessor import Pipe, THRESHOLD

STATS_FILE_PATH = ".//stats_file.txt"
MODEL_PATH = '..//Models//SVC'
CV = 5
N_JOBS = -1
SCORING = 'accuracy'


def main():
    warnings.simplefilter('ignore')
    reader = DbReader()
    pipe = Pipe(reader)

    X_train, y_train, X_test, y_test = reader.get_all_data()

    preprocessor = pipe.get_pipe()

    model = SVMModel()
    params = {
        'model__C': [0.1, 1, 10, 100, 150],
        'model__class_weight': [None, 'balanced'],
        'model__gamma': ['scale', 'auto', 0.01, 0.1, 1, 10],
        'model__shrinking': [True, False],
        'model__decision_function_shape': ['ovo', 'ovr'],
    }

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model.get_model())
    ])

    # FIND BEST PARAMS
    clf = GridSearchCV(estimator=pipeline, param_grid=params, scoring=SCORING, n_jobs=N_JOBS,
                       cv=CV, return_train_score=True, iid=False)
    t_start = time()
    clf.fit(X_train, y_train)
    t_end = time()
    t_dur = t_end - t_start

    # DISPLAY RESULTS
    print(f"SVM:")
    print(f"Best params: {clf.best_params_}")
    print(f"Best score: {clf.best_score_:.{4}}")
    print(f"Worst score: {clf.cv_results_['mean_test_score'].min():.{4}}")
    print(f"Fitting time: {t_dur:{4}.{6}}")

    # SAVE MODEL
    model = clf.best_estimator_
    model.fit(X_train, y_train)
    dump(model, MODEL_PATH)
    print(f"Score on the test data: {clf.score(X_test, y_test):.{4}}")

    # SAVE LOGS TO FILE
    with open(STATS_FILE_PATH, "a") as file:
        file.write(f"\n\nSVM:\n")
        file.write(f"Best params: {clf.best_params_}\n")
        file.write(f"Best score: {clf.best_score_:.{4}}\n")
        file.write(f"Worst score: {clf.cv_results_['mean_test_score'].min():.{4}}\n")
        file.write(f"Score on the test data: {clf.score(X_test, y_test):.{4}}\n")
        file.write(f"Fitting: \n")
        file.write(f"N_JOBS: {N_JOBS:{3}}, THRESHOLD: {THRESHOLD}\t")
        file.write(f"CV: {CV:{3}}, time: {t_dur:{4}.{5}}\n")


if __name__ == "__main__":
    main()
