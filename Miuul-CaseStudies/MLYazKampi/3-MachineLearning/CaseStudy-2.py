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

train = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\Analysis\Miuul-CaseStudies\MLYazKampi\3-MachineLearning\house-price-train.csv")
test = pd.read_csv(r"C:\Users\kdrcn\OneDrive\Masaüstü\Py\Analysis\Miuul-CaseStudies\MLYazKampi\3-MachineLearning\house-price-test.csv")

POps = helpers.PandasOptions()
POps.PrintOptions()
POps.SetOptions(1, 4, 3, 2)

Helper = helpers.HelperFunctions
Helper(train).QuickView()
train_variables = Helper(train).Variables()

train_variables.sort_values("DTypes")


