from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
hgbr=HistGradientBoostingRegressor()
gbr=GradientBoostingRegressor()
adaboost_model = AdaBoostRegressor()

kf=KFold(n_splits=3,shuffle=True,random_state=42)
grid_search_hist = GridSearchCV(estimator=hgbr, param_grid=param_grid_hist, scoring='neg_mean_squared_error', cv=kf)
grid_search_gb = GridSearchCV(estimator=gbr, param_grid=param_grid_gb, scoring='neg_mean_squared_error', cv=kf)

grid_search_hist.fit(x_train_scaled,y_train)


print(f" Best Hyperparameters for HistogramGradientBoostingRegressor: {grid_search_hist.best_params_}")

y_pred = grid_search_hist.predict(x_val_scaled)


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Validation RMSE:", rmse)

