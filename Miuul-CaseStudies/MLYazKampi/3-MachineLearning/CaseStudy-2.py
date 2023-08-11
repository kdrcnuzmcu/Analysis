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

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from ModuleWizard import helpers
import matplotlib
matplotlib.use("Qt5Agg")

train = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\Analysis\Miuul-CaseStudies\MLYazKampi\3-MachineLearning\house-price-train.csv")
test = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\Analysis\Miuul-CaseStudies\MLYazKampi\3-MachineLearning\house-price-test.csv")

POps = helpers.PandasOptions()
POps.PrintOptions()
POps.SetOptions(1, 4, 3, 2)

Helper = helpers.HelperFunctions
Helper(train).QuickView()
train_variables = Helper(train).Variables()
train_variables.sort_values(["DTypes", "Number-Of-Uniques"])
cat_cols, num_cols, cat_but_car = Helper(train).GrabColNames(cat_th=26, verbose=True)

for col in cat_cols:
    Helper(train).CategoricalsByTarget(col, "SalePrice")

cat_cols = [col for col in cat_cols if col not in ["LowQualFinSF", "3SsnPorch", "PoolArea", "MiscVal"]]
num_cols = num_cols + ["LowQualFinSF", "3SsnPorch", "PoolArea", "MiscVal"]

Helper(train[cat_cols]).Variables()
train[cat_cols] = train[cat_cols].fillna("None")

for col in cat_cols:
    Helper(train).CategoricalsByTarget(col, "SalePrice")

# region cat_cols preprocessing
Helper(train).CategoricalsByTarget("MSZoning", "SalePrice", 0.02)
Helper(train).CategoricalsByTarget("LotShape", "SalePrice", 0.03)
Helper(train).CategoricalsByTarget("LandContour", "SalePrice", 0.04)
Helper(train).CategoricalsByTarget("LotConfig", "SalePrice", 0.04)
Helper(train).CategoricalsByTarget("LandSlope", "SalePrice", 0.05)
Helper(train).CategoricalsByTarget("Condition1", "SalePrice", 0.03)

train.loc[train["Condition2"].isin(["PosN", "RRNn", "PosA", "RRAe", "RRAn"]), "Condition2"] = "Near Railroad"
train.loc[train["Condition2"].isin(["Artery", "Feedr"]), "Condition2"] = "Near Street"

train.loc[train["BldgType"] == "2fmCon", "BldgType"] = "1Fam"
train.loc[train["BldgType"].isin(["TwnhsE", "Twnhs"]), "BldgType"] = "Twnhs"

Helper(train).CategoricalsByTarget("RoofStyle", "SalePrice", 0.01)
Helper(train).CategoricalsByTarget("RoofMatl", "SalePrice", 0.01)
Helper(train).CategoricalsByTarget("Exterior1st", "SalePrice", 0.01)
Helper(train).CategoricalsByTarget("Exterior2nd", "SalePrice", 0.017)

train.loc[train["ExterQual"] == "Fa", "ExterQual"] = "TA"

train.loc[train["ExterCond"] == "Ex", "ExterCond"] = "Gd"
train.loc[train["ExterCond"] == "Po", "ExterCond"] = "Fa"

Helper(train).CategoricalsByTarget("Foundation", "SalePrice", 0.01)

train.loc[train["BsmtCond"] == "Po", "BsmtCond"] = "Fa"
train.loc[train["BsmtFinType1"].isin(["GLQ", "ALQ", "BLQ"]), "BsmtFinType1"] = "LQ"
train.loc[train["BsmtFinType2"].isin(["GLQ", "ALQ", "BLQ"]), "BsmtFinType2"] = "LQ"

Helper(train).CategoricalsByTarget("Heating", "SalePrice", 0.02)

train.loc[train["HeatingQC"] == "Po", "HeatingQC"] = "Fa"

train.loc[train["Electrical"].isin(["FuseA", "FuseF", "FuseP"]), "Electrical"] = "Fuse"
train.loc[train["Electrical"] == "Mix", "Electrical"] = "SBrkr"

train.loc[train["Functional"] == "Min1", "Functional"] = "Typ"
train.loc[train["Functional"].isin(["Min2", "Maj1"]), "Functional"] = "Mod"
train.loc[train["Functional"].isin(["Maj2", "Sev"]), "Functional"] = "Bad"

train.loc[train["FireplaceQu"] == "Ex", "FireplaceQu"] = "Gd"
train.loc[train["FireplaceQu"] == "Po", "FireplaceQu"] = "Fa"

Helper(train).CategoricalsByTarget("GarageType", "SalePrice", 0.02)

train.loc[train["GarageQual"] == "Ex", "GarageQual"] = "Gd"
train.loc[train["GarageQual"] == "Po", "GarageQual"] = "Fa"

train.loc[train["GarageCond"] == "Ex", "GarageCond"] = "Gd"
train.loc[train["GarageCond"] == "Po", "GarageCond"] = "Fa"

Helper(train).CategoricalsByTarget("PoolQC", "SalePrice", 0.02)

train.loc[train["Fence"] == "MnWw", "Fence"] = "MnPrv"

Helper(train).CategoricalsByTarget("MiscFeature", "SalePrice", 0.04)
Helper(train).CategoricalsByTarget("SaleType", "SalePrice", 0.04)
Helper(train).CategoricalsByTarget("SaleCondition", "SalePrice", 0.02)

train.loc[train["MSSubClass"].isin([20, 30, 40, 45, 50]), "MSSubClass"] = "Low"
train.loc[train["MSSubClass"].isin([60, 70, 75, 80, 85, 90]), "MSSubClass"] = "Mod"
train.loc[train["MSSubClass"].isin([120, 150, 160, 180, 190]), "MSSubClass"] = "Hig"

train.loc[train["OverallQual"].isin([1, 2, 3, 4]), "OverallQual"] = "Low"
train.loc[train["OverallQual"].isin([5, 6, 7]), "OverallQual"] = "Mod"
train.loc[train["OverallQual"].isin([8, 9, 10]), "OverallQual"] = "Good"

train.loc[train["OverallCond"].isin([1, 2, 3, 4]), "OverallCond"] = "Low"
train.loc[train["OverallCond"].isin([5, 6, 7]), "OverallCond"] = "Mod"
train.loc[train["OverallCond"].isin([8, 9, 10]), "OverallCond"] = "Good"

train.loc[train["BedroomAbvGr"].isin([0, 1]), "BedroomAbvGr"] = "0-1"
train.loc[train["BedroomAbvGr"].isin([2, 3]), "BedroomAbvGr"] = "2-3"
train.loc[train["BedroomAbvGr"].isin([4, 5]), "BedroomAbvGr"] = "4-5"
train.loc[train["BedroomAbvGr"].isin([6, 7, 8]), "BedroomAbvGr"] = "6-7-8"

train.loc[train["TotRmsAbvGrd"].isin([0, 1, 2, 3]), "TotRmsAbvGrd"] = "0-3"
train.loc[train["TotRmsAbvGrd"].isin([4, 5, 6, 7]), "TotRmsAbvGrd"] = "4-7"
train.loc[train["TotRmsAbvGrd"].isin([8, 9, 10, 11]), "TotRmsAbvGrd"] = "8-11"
train.loc[train["TotRmsAbvGrd"].isin([12, 13, 14]), "TotRmsAbvGrd"] = "12-14"

train.loc[train["GarageCars"].isin([1, 2]), "GarageCars"] = "1-2"
train.loc[train["GarageCars"].isin([3, 4]), "GarageCars"] = "3-4"

train.loc[train["MoSold"].isin([1, 2, 3]), "MoSold"] = "1-3"
train.loc[train["MoSold"].isin([4, 5, 6]), "MoSold"] = "4-6"
train.loc[train["MoSold"].isin([7, 8, 9]), "MoSold"] = "7-9"
train.loc[train["MoSold"].isin([10, 11, 12]), "MoSold"] = "10-12"

train["YearBuilt"] = pd.cut(train["YearBuilt"], bins=[1870, 1900, 1920, 1940, 1960, 1980, 2000, 2010])
train["GarageYrBlt"] = pd.cut(train["GarageYrBlt"], bins=[1898, 1920, 1940, 1960, 1980, 2000, 2010])
train["MiscVal"] = pd.cut(train["MiscVal"], bins=[-1, 1, 499, 999, 1999, 20000])
train["3SsnPorch"] = pd.cut(train["3SsnPorch"], bins=[-1, 1, 149, 199, 299, 999])
train["ScreenPorch"] = pd.cut(train["ScreenPorch"], bins=[-1, 1, 99, 199, 399, 499])
train["LotFrontage"] = pd.cut(train["LotFrontage"], bins=[0, 40, 80, 120, 160, 200, 300, 400])
train["MasVnrArea"] = pd.cut(train["MasVnrArea"], bins=[-1, 1, 40, 80, 160, 320, 720, 1440, 2880])
train["GarageArea"] = pd.cut(train["GarageArea"], bins=[-1, 80, 160, 320, 640, 1280, 2560])
# endregion

for col in cat_cols:
    Helper(train).CategoricalsByTarget(col, "SalePrice")

Helper(train).Variables()

train["MiscVal"]

train["GarageYrBlt"].min()
train["GarageYrBlt"].max()

train.loc[1453]

plt.hist(train["GarageArea"])
temp = train["GarageArea"].value_counts()
temp.reset_index().sort_values("GarageArea")





LOF = LocalOutlierFactor()
LOF.fit_predict(train[num_cols].drop("SalePrice", axis=1))


