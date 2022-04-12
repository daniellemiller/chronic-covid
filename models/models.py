import os

import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ML packages
from sklearn import metrics



##### Models interface ######

class Model():
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
        self.test_shap = None
        self.mdl_shap = None


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

    def model_shap(self):
        X = self.X.drop(columns=['patient'])
        y = self.y
        try:
            _ = self.model_fit(X, X, y)
            explainer = shap.TreeExplainer(self.clf)
            shap_values = explainer.shap_values(X)
            self.mdl_shap = shap_values
        except:
            return

    def split_and_classify(self):
        reports, feature_importances, list_test_shap_values= [], [], []

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
            try:
                explainer = shap.TreeExplainer(self.clf)
                test_shap = explainer.shap_values(X_test)
                list_test_shap_values.append(test_shap)
            except:
                continue

        self.report = pd.concat(reports)
        self.importances = pd.concat(feature_importances)

        if list_test_shap_values != []:
            shap_values_test = np.array(list_test_shap_values[0])
            for i in range(1,len(self.folds)):
                shap_values_test = np.concatenate((shap_values_test,np.array(list_test_shap_values[i])),axis=1)
            self.test_shap = shap_values_test

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
        order = importances.groupby('feature_name')['importance'].mean().sort_values(ascending=False).index
        with sns.plotting_context("talk"):
            ax, fig = plt.subplots(figsize=(10,5))
            sns.barplot(x='importance', y='feature_name', data=importances, palette='Oranges', order=order)
            _ = plt.xticks(rotation=65)
            plt.xlabel('')
            plt.ylabel("Importance")
            plt.savefig(os.path.join(data_dir, 'importance.pdf'), bbox_inches='tight')
            plt.clf()

    def plot_shap_stats(self):
        mdl_shape = self.mdl_shap
        test_shap = self.test_shap
        if mdl_shape is None or test_shap is None:
            return
        data_dir = os.path.join(self.out_dir, self.name)
        os.makedirs(data_dir, exist_ok=True)
        X = self.X.drop(columns=['patient'])

        shap.summary_plot(mdl_shape[1], X, plot_type='bar', show=False)
        plt.savefig(os.path.join(data_dir, f'{self.name}_shap_mean.pdf'), bbox_inches='tight')
        plt.clf()

        shap.summary_plot(mdl_shape[1], X, show=False)
        plt.savefig(os.path.join(data_dir, f'{self.name}_shap.pdf'), bbox_inches='tight')
        plt.clf()

        shap.summary_plot(test_shap[1], X, plot_type='bar', show=False)
        plt.savefig(os.path.join(data_dir, f'LOPOCV_{self.name}_shap_mean.pdf'), bbox_inches='tight')
        plt.clf()

        shap.summary_plot(test_shap[1], X, show=False)
        plt.savefig(os.path.join(data_dir, f'LOPOCV_{self.name}_shap.pdf'), bbox_inches='tight')
        plt.clf()



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
        self.model_shap()
        self.plot_shap_stats()
        self.wrap_up()

