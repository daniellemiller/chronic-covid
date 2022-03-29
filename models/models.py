import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ML packages
from sklearn import metrics



##### Models interface ######

class Model(object):
    def __init__(self, X, y, out_dir, clf=None, alias='CLF', fold_id='patient', target='Ab evasion'):
        self.X = X
        self.y = y
        self.clf = clf
        self.out_dir = out_dir
        self.folds = X[fold_id].unique()
        self.target = target
        self.name = alias
        self.report = None
        self.importances = None
        self.acc = None
        self.w_acc = None


    def model_fit(self, X_train, X_test, y_train):
        clf = self.clf
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        self.clf = clf
        return predicted

    def LOPOCV(self, patient):
        X = self.X
        y = self.y
        target = self.target

        data = X.copy()
        data[target] = y

        patient_data = data[data['patient'] == patient].drop(columns=['patient'])
        train_data = data[data['patient'] != patient].drop(columns=['patient'])

        return train_data.drop(columns=[target]), train_data[target], patient_data.drop(columns=[target]), patient_data[target]

    def split_and_classify(self):
        reports, feature_importances = [], []

        for fold in self.folds:
            X_train, y_train, X_test, y_test = self.LOPOCV(patient=fold)
            predicted = self.model_fit(X_train, X_test, y_train)

            stats = pd.DataFrame(metrics.classification_report(y_test, predicted, output_dict=True,
                                                               zero_division=0)).T
            stats['patient'] = fold

            try:
                importances = self.clf.feature_importances_
            except:
                importances = self.clf.coef_.T
            mdl_importances = pd.DataFrame(importances, index=X_test.columns,
                                  columns=['importance']).reset_index().rename(columns={"index":'feature_name'})
            mdl_importances['patient'] = fold

            reports.append(stats)
            feature_importances.append(mdl_importances)

        self.report = pd.concat(reports)
        self.importances = pd.concat(feature_importances)
        return self.report, self.importances

    def summarize_accuracy(self):
        report = self.report
        # calculate weighted accuracy
        pr = report.loc['weighted avg']
        acc = report.loc['accuracy'][['precision','patient']].reset_index(drop=True).rename(columns={'precision':'accuracy'})
        acc = pd.merge(pr,acc, on='patient')

        for c in ["precision","recall","f1-score", "accuracy"]:
            acc[f'w_{c}'] = acc[c] * acc['support']
        w_acc = pd.DataFrame(acc[["w_precision","w_recall","w_f1-score", "w_accuracy"]].sum() / acc['support'].sum()).T
        w_acc['mdl'] = self.name
        acc['mdl'] = self.name

        self.acc = acc
        self.w_acc = w_acc
        return acc, w_acc

    def plot_importance_bar(self):
        importances = self.importances
        data_dir = os.path.join(self.out_dir, self.name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        with sns.plotting_context("talk"):
            ax, fig = plt.subplots(figsize=(10,5))
            sns.barplot(x='feature_name', y='importance', data=importances, palette='Oranges')
            _ = plt.xticks(rotation=65)
            plt.xlabel('')
            plt.ylabel("Importance")
            plt.savefig(os.path.join(data_dir, 'importance.png'), dpi=300, bbox_inches='tight')


    def wrap_up(self):
        q_dir = os.path.join(self.out_dir, self.name)
        if not os.path.exists(q_dir):
            os.makedirs(q_dir)

        self.report.to_csv(os.path.join(q_dir, f"{self.name}_report.csv"), index=False)
        self.importances.to_csv(os.path.join(q_dir, f"{self.name}_importances.csv"), index=False)
        self.acc.to_csv(os.path.join(q_dir, f"{self.name}_accuracy.csv"), index=False)
        self.w_acc.to_csv(os.path.join(q_dir, f"{self.name}_weighted_accuracy.csv"), index=False)
        print("Done")


    def classification_pipeline(self):
        self.split_and_classify()
        self.summarize_accuracy()
        self.plot_importance_bar()
        self.wrap_up()

