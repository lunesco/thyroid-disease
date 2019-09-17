import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class Plotter:
    def __init__(self, reader):
        self.reader = reader

    def heatmap(self):
        sns.heatmap(self.reader.get_X().corr(), annot=True)
        plt.show()

    def kdeplot(self):
        full_data = self.reader.get_X()
        sns.kdeplot(full_data['sex'], label="sex")
        sns.kdeplot(full_data['age'], label="age")
        sns.kdeplot(full_data['TSH'], label="TSH")
        plt.legend()
        plt.show()

    def pairplot(self):
        features = ['sex', 'referral_source', 'age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        X = reader.get_X()
        sns.pairplot(X, vars=features, hue='diagnosis')
        plt.show()

    def correlation_matrix(self, data):
        plt.matshow(data.corr())
        plt.show()

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        classes = ['hyper', 'hypo', 'negative']
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title="Confusion matrix",
               ylabel="True label",
               xlabel="Predicted label")
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                print(f"cm[{i}][{j}] = {cm[i][j]}")
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.subplots_adjust(hspace=.001)
        plt.show()

    def classification_report(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        return classification_report(y_true, y_pred)
