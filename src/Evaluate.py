from joblib import load
from sklearn.pipeline import Pipeline

from src.DbReader import DbReader
from src.Preprocessor import Pipe
from src.Plotter import Plotter

MODEL_PATH = '..//Models//SVC'
STATS_FILE_PATH = './/stats_file.txt'


def main():
    reader = DbReader()
    plotter = Plotter(reader)
    X_test, y_test = reader.get_test_data()
    preprocessor = Pipe(reader).get_pipe()

    model = load(MODEL_PATH)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    print(model.score(X_test, y_test))

    # SAVE THE RESULTS
    y_pred = model.predict(X_test)
    with open(STATS_FILE_PATH, "a") as file:
        file.write(plotter.classification_report(y_test, y_pred))

    # plotter.confusion_matrix(y_test, model.predict(X_test))
    # plotter.correlation_matrix(X_test)


if __name__ == "__main__":
    main()
