
from feature_Extraction import feature_extraction
test_features=feature_extraction(test_logs)
test_features.isnull().sum()
test_features.replace([np.inf, -np.inf], np.nan, inplace=True)

test_features = test_features.dropna(axis=1, how='any')

X_test=test_features.drop("id",axis=1)

common_features=set(x_train.columns) & set(X_test.columns)

common_features=list(common_features)
X_test_common=X_test[common_features]
X_test_common = X_test_common[x_train.columns]
X_test_imputed=imputer.transform(X_test_common)

X_test_common.dropna(axis=1, how='any')

X_test_scaler=scaler.transform(X_test_imputed)

test_predictions=inverted_adaboost.predict(X_test_scaler)


# Flatten nested lists and ensure that the length matches the number of unique IDs
test_predictions_flat = [item for sublist in test_predictions for item in sublist]
test_predictions_flat = test_predictions_flat[:len(test_logs["id"])]

# Create the DataFrame with flattened predictions
# Replace NaN values with a default value (e.g., 0.0)
test_predictions_flat_no_nan = [score if not np.isnan(score) else 0.0 for score in test_predictions_flat]

# Round predicted scores to the nearest half
rounded_predictions = [round(score * 2) / 2 for score in test_predictions_flat_no_nan]

# Use zip and pd.Series to create the DataFrame
result_df = pd.DataFrame({"id": test_logs["id"], "predicted_score": pd.Series(rounded_predictions)})

# Save the predictions to a CSV file
result_df.to_csv("/kaggle/working/predictions.csv", index=False)

result_df.head()

