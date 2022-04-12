import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

from models import Model

names = [
    "Linear_SVM",
    "Decision_Tree",
    "Random_Forest",
    "Logistic_Regression",
]

classifiers = [
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=3),
    RandomForestClassifier(max_depth=3, n_estimators=500),
    LogisticRegression(),
]

df = pd.read_csv("data/model_data.csv")
features = ['patient', 'Day of sequencing', 'CoV Ab treatment',
             'Viral rebound','Steroids treatment', 'Inferred B cell depletion',
             'Sex', 'Age', 'Ab evasion']

data = df[features]
X = data.drop(columns=['Ab evasion'])
y = data['Ab evasion']

for name, clf in tqdm(zip(names, classifiers)):
    mdl = Model(X, y, "./stats/", clf=clf,
                alias=name, fold_id='patient', target='Ab evasion')
    mdl.classification_pipeline()
