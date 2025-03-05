import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

param_grid_hist={
    'learning_rate': [0.01, 0.1, 0.2],
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 2, 4,3,9]
}

param_grid_gb={
    'learning_rate': [0.01, 0.1, 0.2,0.001],
    'n_estimators': [60, 100, 200,250],
    'max_depth': [3, 5, 7,10,11,9,4,2,13,20],
    'min_samples_leaf': [1, 2, 4,3,9,5,7,6,4,10,20]
}
param_grid_adaboost = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.001]
}



final_df=new_data.merge(train_scores,on="id")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
x=final_df.drop(["id","score"],axis=1)
y=final_df["score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


imputer = SimpleImputer(strategy='median')
x_train_imputed = imputer.fit_transform(x_train)
x_val_imputed = imputer.transform(x_test)




scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_val_scaled = scaler.transform(x_val_imputed)


