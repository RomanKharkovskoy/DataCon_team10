import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class MainModel:
    def __init__(self):

        cleaned_df = pd.read_csv("data\\cleaned_data.csv", index_col=0) 

        labelencoder = LabelEncoder() 
        self.encoded_df = cleaned_df.copy()
        for column in self.encoded_df.select_dtypes(include=object).columns:
            self.encoded_df[column]= labelencoder.fit_transform(self.encoded_df[column])

        X = self.encoded_df.drop(columns=['zoi_drug_np'])
        y = self.encoded_df['zoi_drug_np']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25)

        self.best_params = json.load(open('data\\best_params_regressions.json', 'r'))

        self.get_predictions()

    def get_predictions(self):

        self.gradient_boosting_regressor_model = GradientBoostingRegressor(
            max_depth=self.best_params['gradient_boosting']['max_depth'],
            max_features=self.best_params['gradient_boosting']['max_features'],
            n_estimators=self.best_params['gradient_boosting']['n_estimators']
        ).fit(self.X_train, self.y_train)

        y_pred_gradient_boosting = self.gradient_boosting_regressor_model.predict(self.X_test)

        rmse_gradient_boosting = mean_squared_error(self.y_test, y_pred_gradient_boosting, squared = False)
        r2_gradient_boosting = r2_score(self.y_test, y_pred_gradient_boosting)
    
    def draw_plot(self):

        plt.figure(figsize=(10, 10))
        features = self.gradient_boosting_regressor_model.feature_importances_
        self.encoded_df.rename(columns={"drug_class_drug_bank":"drug_class"}, inplace=True)
        labels = self.encoded_df.drop(columns=['zoi_drug_np']).columns

        sns.barplot(x=labels, y=features)
        plt.xticks(rotation=90)
        plt.show()

def main():
    main_model = MainModel()
    main_model.draw_plot()

if __name__ == "__main__":
    main()