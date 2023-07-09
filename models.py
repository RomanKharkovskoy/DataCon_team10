import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class Models:
    def __init__(self):

        cleaned_df = pd.read_csv("data\\cleaned_data.csv", index_col=0) 

        labelencoder = LabelEncoder() 
        encoded_df = cleaned_df.copy()
        for column in encoded_df.select_dtypes(include=object).columns:
            encoded_df[column]= labelencoder.fit_transform(encoded_df[column])

        X = encoded_df.drop(columns=['zoi_drug_np'])
        y = encoded_df['zoi_drug_np']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25)

        self.best_params = json.load(open('data\\best_params_regressions.json', 'r'))

        self.teach_models()
        self.get_predictions()
        self.get_errors()

    def teach_models(self):

        self.linear_regression_model = LinearRegression().fit(self.X_train, self.y_train)

        self.elastic_net_model = ElasticNet(
            alpha=self.best_params['elastic_net']['alpha']
        ).fit(self.X_train, self.y_train)

        self.ridge_model = Ridge(
            alpha=self.best_params['ridge']['alpha']
        ).fit(self.X_train, self.y_train)

        self.lasso_model = Lasso(
            alpha=self.best_params['lasso']['alpha']
        ).fit(self.X_train, self.y_train)

        self.svr_model = SVR(
            C=self.best_params['svr']['C'],
            coef0=self.best_params['svr']['coef0'],
            degree=self.best_params['svr']['degree'],
            gamma=self.best_params['svr']['gamma'],
            kernel=self.best_params['svr']['kernel']
        ).fit(self.X_train, self.y_train)

        self.decision_tree_regressor_model = DecisionTreeRegressor(
            max_depth=self.best_params['decision_tree']['max_depth'],
            max_leaf_nodes=self.best_params['decision_tree']['max_leaf_nodes'],
            min_samples_leaf=self.best_params['decision_tree']['min_samples_leaf'],
            min_weight_fraction_leaf=self.best_params['decision_tree']['min_weight_fraction_leaf']
        ).fit(self.X_train, self.y_train)

        self.random_forest_regressor_model = RandomForestRegressor(
            max_depth=self.best_params['random_forest']['max_depth'],
            max_features=self.best_params['random_forest']['max_features'],
            max_leaf_nodes=self.best_params['random_forest']['max_leaf_nodes'],
            n_estimators=self.best_params['random_forest']['n_estimators']
        ).fit(self.X_train, self.y_train)

        self.gradient_boosting_regressor_model = GradientBoostingRegressor(
            max_depth=self.best_params['gradient_boosting']['max_depth'],
            max_features=self.best_params['gradient_boosting']['max_features'],
            n_estimators=self.best_params['gradient_boosting']['n_estimators']
        ).fit(self.X_train, self.y_train)

        self.ada_boost_regressor_model = AdaBoostRegressor(
            learning_rate=self.best_params['ada_boost']['learning_rate'],
            n_estimators=self.best_params['ada_boost']['n_estimators']
        ).fit(self.X_train, self.y_train)

    def get_predictions(self):

        self.y_pred_linear = self.linear_regression_model.predict(self.X_test)
        self.y_pred_elastic_net = self.elastic_net_model.predict(self.X_test)
        self.y_pred_ridge = self.ridge_model.predict(self.X_test)
        self.y_pred_lasso = self.lasso_model.predict(self.X_test)
        self.y_pred_svr = self.svr_model.predict(self.X_test)
        self.y_pred_decision_tree = self.decision_tree_regressor_model.predict(self.X_test)
        self.y_pred_random_forest = self.random_forest_regressor_model.predict(self.X_test)
        self.y_pred_gradient_boosting = self.gradient_boosting_regressor_model.predict(self.X_test)
        self.y_pred_ada_boost = self.ada_boost_regressor_model.predict(self.X_test)

    def get_errors(self):

        self.rmse_linear = mean_squared_error(self.y_test, self.y_pred_linear, squared = False)
        self.rmse_elastic_net = mean_squared_error(self.y_test, self.y_pred_elastic_net, squared = False)
        self.rmse_ridge = mean_squared_error(self.y_test, self.y_pred_ridge, squared = False)
        self.rmse_lasso = mean_squared_error(self.y_test, self.y_pred_lasso, squared = False)
        self.rmse_svr = mean_squared_error(self.y_test, self.y_pred_svr, squared = False)
        self.rmse_decision_tree = mean_squared_error(self.y_test, self.y_pred_decision_tree, squared = False)
        self.rmse_random_forest = mean_squared_error(self.y_test, self.y_pred_random_forest, squared = False)
        self.rmse_gradient_boosting = mean_squared_error(self.y_test, self.y_pred_gradient_boosting, squared = False)
        self.rmse_ada_boost = mean_squared_error(self.y_test, self.y_pred_ada_boost, squared = False)

        self.r2_linear = r2_score(self.y_test, self.y_pred_linear)
        self.r2_elastic_net = r2_score(self.y_test, self.y_pred_elastic_net)
        self.r2_ridge = r2_score(self.y_test, self.y_pred_ridge)
        self.r2_lasso = r2_score(self.y_test, self.y_pred_lasso)
        self.r2_svr = r2_score(self.y_test, self.y_pred_svr)
        self.r2_decision_tree = r2_score(self.y_test, self.y_pred_decision_tree)
        self.r2_random_forest = r2_score(self.y_test, self.y_pred_random_forest)
        self.r2_gradient_boosting = r2_score(self.y_test, self.y_pred_gradient_boosting)
        self.r2_ada_boost = r2_score(self.y_test, self.y_pred_ada_boost)
    
    def draw_plot(self):

        labels = ['LinearRegression', 'ElasticNet', 'Ridge', 'Lasso', 'SVR', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'AdaBoost']
        rmse = [self.rmse_linear, self.rmse_elastic_net, self.rmse_ridge, self.rmse_lasso, self.rmse_svr, self.rmse_decision_tree, self.rmse_random_forest, self.rmse_gradient_boosting, self.rmse_ada_boost] 
        r2 = [self.r2_linear, self.r2_elastic_net, self.r2_ridge, self.r2_lasso, self.r2_svr, self.r2_decision_tree, self.r2_random_forest, self.r2_gradient_boosting, self.r2_ada_boost]

        df_labels_errors = pd.DataFrame({
            'labels': labels,
            'RMSE': rmse,
            'R2': r2
        })

        df_melted = df_labels_errors.melt(id_vars='labels', var_name='metric', value_name='value')

        plt.figure(figsize=(20, 25))
        sns.barplot(x="labels", y="value", hue="metric", data=df_melted)
        plt.xlabel("Модели")
        plt.ylabel("Ошибка")
        plt.title("RMSE и R2 для различных моделей")
        plt.legend(title="Метрика", loc="upper right")
        plt.show()

def main():
    models = Models()
    models.draw_plot()

if __name__ == "__main__":
    main()