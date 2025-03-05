new_data=pd.read_csv("/kaggle/temp/trimmed_features.csv")
#new_data.head()
#new_data = new_data.drop(["event_rate", "text_change_ratio", "word_count_change_rate"], axis=1)
new_data.head()



new_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns with NaN or inf values
new_data = new_data.dropna(axis=1, how='any')

# Display columns with NaN values (if any)



