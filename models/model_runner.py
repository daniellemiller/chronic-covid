import pandas as pd

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm

from models import Model

names = [
    "Linear_SVM",
    "Decision_Tree",
    "Random_Forest",
    "AdaBoost",
    "Logistic_Regression",
    "XGBoost"
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=3, n_estimators=500),
    AdaBoostClassifier(),
    LogisticRegression(),
    XGBClassifier(),
]

df = pd.read_csv("/Volumes/GoogleDrive/Shared drives/Lab.Danielle/Projects/chronic_covid/model_data.csv")
features = ['patient', 'Disease progression', 'CoV Ab treatment',
             'Viral rebound','Steroids treatment', 'Inferred B cell depletion',
             'Sex', 'Age', 'Ab evasion']

data = df[features]
X = data.drop(columns=['Ab evasion'])
y = data['Ab evasion']

for name, clf in tqdm(zip(names, classifiers)):
    mdl = Model(X, y, "/Volumes/GoogleDrive/Shared drives/Lab.Danielle/Projects/chronic_covid/stats", clf=clf,
                alias=name, fold_id='patient', target='Ab evasion')
    mdl.classification_pipeline()
