import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import settings.hyperparametrs as hp     

warnings.filterwarnings('ignore')


class CrossValidation:
    def __init__(self):
        self.cleaned_df = pd.read_csv("data\\cleaned_data.csv", index_col=0) 

        self.labelencoder = LabelEncoder() 
        self.encoded_df = self.cleaned_df.copy()
        for column in self.encoded_df.select_dtypes(include=object).columns:
            self.encoded_df[column]= self.labelencoder.fit_transform(self.encoded_df[column])

        self.X = self.encoded_df.drop(columns=['zoi_drug_np'])
        self.y = self.encoded_df['zoi_drug_np']

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)

class CVElasticNet(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        elastic_net_search_model = ElasticNetCV(
            l1_ratio = hp.elastic_net_l1_ratio,
            n_alphas = hp.elastic_net_n_alphas,
            cv = hp.cv
        )
        elastic_net_search_model.fit(self.X_train, self.y_train)

        self.elastic_net_best_params = {
            'alpha': elastic_net_search_model.alpha_,
            'l1_ratio': elastic_net_search_model.l1_ratio_
        }

        return self.elastic_net_best_params

class CVRidge(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        ridge_search_model = RidgeCV(
            alphas = hp.ridge_alphas,
            cv = hp.cv,
            scoring = hp.score
        )
        ridge_search_model.fit(self.X_train, self.y_train)

        self.ridge_best_params = {
            'alpha': ridge_search_model.alpha_
        }

        return self.ridge_best_params

class CVLasso(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        lasso_search_model = LassoCV(
            alphas = hp.lasso_alphas,
            cv = hp.cv
        )
        lasso_search_model.fit(self.X_train, self.y_train)

        self.lasso_best_params = {
            'alpha': lasso_search_model.alpha_
        }

        return self.lasso_best_params

class CVSVR(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        svr_model = SVR()
        SVR_search_model = GridSearchCV(
            estimator = svr_model,
            param_grid = hp.SVR_hyperparametrs,
            cv = hp.cv,
            scoring = hp.score
        )

        SVR_search_model.fit(self.X_train, self.y_train)

        return SVR_search_model.best_params_

class CVDecisionTreeRegressor(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        decision_tree_regressor_model = DecisionTreeRegressor()

        decision_tree_grid_search = GridSearchCV(
            estimator = decision_tree_regressor_model,
            param_grid = hp.decision_tree_regressor_hyperparametrs,
            cv = hp.cv,
            scoring = hp.score
        )

        decision_tree_grid_model = decision_tree_grid_search.fit(self.X_train, self.y_train)

        return decision_tree_grid_model.best_params_

class CVRandomForestRegressor(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        random_forest_regressor_model = RandomForestRegressor()

        random_forest_grid_search = GridSearchCV(
            estimator = random_forest_regressor_model,
            param_grid = hp.random_forest_hyperparametrs,
            cv = hp.cv,
            scoring = hp.score
        )

        random_forest_grid_model = random_forest_grid_search.fit(self.X_train, self.y_train)

        return random_forest_grid_model.best_params_

class CVGradientBoostingRegressor(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        gradient_boosting_regressor_model = GradientBoostingRegressor()

        gradient_boosting_grid_search = GridSearchCV(
            estimator = gradient_boosting_regressor_model,
            param_grid = hp.gradient_boosting_hyperparametrs,
            scoring = hp.score,
            cv = hp.cv
        )

        gradient_boosting_grid_model = gradient_boosting_grid_search.fit(self.X_train, self.y_train)

        return gradient_boosting_grid_model.best_params_

class CVAdaBoostRegressor(CrossValidation):
    def __init__(self):
        super().__init__()

    def get_best_params(self):
        ada_boost_regressor_model = AdaBoostRegressor()

        ada_boost_regressor_grid_search = GridSearchCV(
            estimator = ada_boost_regressor_model,
            param_grid = hp.ada_boost_regressor_hyperparametrs,
            scoring = hp.score,
            cv = hp.cv
        )

        ada_boost_regressor_grid_model = ada_boost_regressor_grid_search.fit(self.X_train, self.y_train)

        return ada_boost_regressor_grid_model.best_params_


class Data():
    def __init__(self):
        self.cv_regressors = {
            'elastic_net': CVElasticNet().get_best_params(),
            'ridge': CVRidge().get_best_params(),
            'lasso': CVLasso().get_best_params(),
            'svr': CVSVR().get_best_params(),
            'decision_tree': CVDecisionTreeRegressor().get_best_params(),
            'random_forest': CVRandomForestRegressor().get_best_params(),
            'gradient_boosting': CVGradientBoostingRegressor().get_best_params(),
            'ada_boost': CVAdaBoostRegressor().get_best_params()
        }

    def get_results(self):
        with open('data\\best_params_regressions.json', 'w') as fp:
            json.dump(self.cv_regressors, fp)

if __name__ == "__main__":
    data = Data()
    data.get_results()
    print("Ready")
    
