import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, make_scorer, r2_score
from sklearn.neighbors import LocalOutlierFactor
from ModuleWizard import helpers
import matplotlib
matplotlib.use("Qt5Agg")

hitters = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\Analysis\Miuul-CaseStudies\MLYazKampi\3-MachineLearning\hitters.csv")

MAPE_scorer = make_scorer(mean_absolute_percentage_error)

POps = helpers.PandasOptions()
POps.PrintOptions()
POps.SetOptions(1, 4, 3)

Helper = helpers.HelperFunctions
Helper(hitters).QuickView()
Helper(hitters).Variables()

cat_cols, num_cols, cat_but_car = Helper(hitters).GrabColNames()
num_cols
cat_cols

X = pd.get_dummies(hitters, columns=cat_cols, drop_first=True, prefix="ENC", dtype=float)

Helper(X).Variables()
for col in num_cols:
    Helper(X).Outliers(col, low_Quantile=0.05, high_Quantile=0.95)

LOF = LocalOutlierFactor()
LOF.fit_predict(X.drop("Salary", axis=1))
X_scores = LOF.negative_outlier_factor_
scores = pd.DataFrame(np.sort(X_scores))
scores.plot(stacked=True, xlim=[0, 30], style=".-")
th = np.sort(X_scores)[3]

outlier_index = X[X_scores < th].index
X = X.drop(outlier_index, axis=0)

train = X[X["Salary"].notnull()]
Xtrain = train.drop("Salary", axis=1)
ytrain = train["Salary"]

test = X[X["Salary"].isnull()]

LR = LinearRegression()
L = Lasso()
R = Ridge(alpha=0.1)
EN = ElasticNet()

LR_cvs = cross_val_score(LR, Xtrain, ytrain, cv=10, scoring=MAPE_scorer)
L_cvs = cross_val_score(L, Xtrain, ytrain, cv=10, scoring=MAPE_scorer)
R_cvs = cross_val_score(R, Xtrain, ytrain, cv=10, scoring=MAPE_scorer)
EN_cvs = cross_val_score(EN, Xtrain, ytrain, cv=10, scoring=MAPE_scorer)

np.std(LR_cvs)
np.std(L_cvs)
np.std(R_cvs)
np.std(EN_cvs)

num_cols.remove("Salary")
# BASE MODEL # BASE MODEL # BASE MODEL # BASE MODEL # BASE MODEL # BASE MODEL # BASE MODEL

for col in num_cols:
    Helper(Xtrain).Scaler(col, "minmax")

X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)

L.fit(X_train, y_train)
L.score(X_train, y_train)
# 0.5234589987212075
L.score(X_test, y_test)
# 0.6663771148848616

# NEW MODEL VOL 2 # NEW MODEL VOL 2 # NEW MODEL VOL 2 # NEW MODEL VOL 2 # NEW MODEL VOL 2
train = X[X["Salary"].notnull()]
Xtrain = train.drop("Salary", axis=1)
ytrain = train["Salary"]
for col in num_cols:
    Helper(Xtrain).Scaler(col, "robust")
X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)

L = Lasso(alpha=0.1)
L.fit(X_train, y_train)
L.score(X_train, y_train)
# 0.5512027164291327
L.score(X_test, y_test)
# 0.6318285623277013

# NEW MODEL VOL 3 # NEW MODEL VOL 3 # NEW MODEL VOL 3 # NEW MODEL VOL 3 # NEW MODEL VOL 3
train = X[X["Salary"].notnull()]
Xtrain = train.drop("Salary", axis=1)
ytrain = train["Salary"]
for col in num_cols:
    Helper(Xtrain).Scaler(col, "robust")
X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)

L = Lasso(alpha=1)
L.fit(X_train, y_train)
L.score(X_train, y_train)
# 0.5457609737595883
L.score(X_test, y_test)
# 0.6518378491371086

# NEW MODEL VOL 4 # NEW MODEL VOL 4 # NEW MODEL VOL 4 # NEW MODEL VOL 4 # NEW MODEL VOL 4

train = X[X["Salary"].notnull()]
Xtrain = train.drop("Salary", axis=1)
ytrain = train["Salary"]

def feature_ext(Xtrain):
    Xtrain["New_AtBatRatio"] = Xtrain["AtBat"] / Xtrain["CAtBat"] * 100
    Xtrain["New_HitsRatio"] = Xtrain["Hits"] / Xtrain["CHits"] * 100
    Xtrain["New_HmRunRatio"] = Xtrain["HmRun"] / Xtrain["CHmRun"] * 100
    Xtrain["New_RBIRatio"] = Xtrain["RBI"] / Xtrain["CRBI"] * 100
    Xtrain["New_Walks"] = Xtrain["Walks"] / Xtrain["CWalks"] * 100
    Xtrain["New_RunsRatio"] = Xtrain["Runs"] / Xtrain["CRuns"] * 100

    Xtrain["New_AtBatRatio_byYear"] = Xtrain["CAtBat"] / Xtrain["Years"]
    Xtrain["New_HitsRatio_byYear"] = Xtrain["CHits"] / Xtrain["Years"]
    Xtrain["New_HmRunRatio_byYear"] = Xtrain["CHmRun"] / Xtrain["Years"]
    Xtrain["New_RBIRatio_byYear"] = Xtrain["CRBI"] / Xtrain["Years"]
    Xtrain["New_Walks_byYear"] = Xtrain["CWalks"] / Xtrain["Years"]
    Xtrain["New_RunsRatio_byYear"] = Xtrain["CRuns"] / Xtrain["Years"]

    Xtrain["New_AtBatSeason"] = Xtrain["New_AtBatRatio"] - Xtrain["New_AtBatRatio_byYear"]
    Xtrain["New_HitsSeason"] = Xtrain["New_HitsRatio"] - Xtrain["New_HitsRatio_byYear"]
    Xtrain["New_HmRunSeason"] = Xtrain["New_HmRunRatio"] - Xtrain["New_HmRunRatio_byYear"]
    Xtrain["New_RBISeason"] = Xtrain["New_RBIRatio"] - Xtrain["New_RBIRatio_byYear"]
    Xtrain["New_WalksSeason"] = Xtrain["New_Walks"] - Xtrain["New_Walks_byYear"]
    Xtrain["New_RunsSeason"] = Xtrain["New_RunsRatio"] - Xtrain["New_RunsRatio_byYear"]

    Xtrain["New_AllRatios"] = Xtrain["New_AtBatRatio"] * \
                              Xtrain["New_HitsRatio"] * \
                              Xtrain["New_HmRunRatio"] * \
                              Xtrain["New_RBIRatio"] * \
                              Xtrain["New_Walks"] * \
                              Xtrain["New_RunsRatio"]
    return Xtrain

Xtrain = feature_ext(Xtrain)

for col in Xtrain.columns:
    Helper(Xtrain).Scaler(col, "robust")

nans = Xtrain[Xtrain["New_HmRunRatio"].isnull()].index
Xtrain = Xtrain.drop(nans)
ytrain = ytrain.drop(nans)

X_train, X_test, y_train, y_test = train_test_split(Xtrain, ytrain, test_size=0.2, random_state=42)

L = Lasso()

L.fit(X_train, y_train)
L.score(X_train, y_train)
# 0.7207599008819809
L.score(X_test, y_test)
# 0.7091568231238095

test = test.drop("Salary", axis=1)

test = feature_ext(test)

for col in test.columns:
    Helper(test).Scaler(col, "robust")

test = test.dropna()
L.predict(test)
