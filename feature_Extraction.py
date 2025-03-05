from tqdm import tqdm
def feature_extraction(data):
    
    data["time_interval"]=data.groupby("id")["action_time"].diff()
    data["event_rate"]=1/data["time_interval"]
    
    data["event_duration_variation"]=data.groupby("id")["action_time"].transform("std")
    data['event_sequence_change'] = (data['down_event'] != data['down_event'].shift(1)).astype(int)

    # Calculate 'total_characters' as the cumulative sum of non-Backspace events
    data['total_characters'] = (data['down_event'] != 'Backspace').groupby(data['id']).cumsum()

    data["event_transition_count"]=data.groupby(["id","down_event","up_event"]).cumcount()+1
    data['total_characters'] = data['total_characters'].replace(0, np.nan)
    data['down_event_numeric'] = pd.factorize(data['down_event'])[0]

# Calculate 'unique_events_count' by dividing the number of unique events by 'total_characters'
    data["unique_events_count"] = data.groupby("id")["down_event_numeric"].transform('nunique') / data["total_characters"]

    
    # Text change features
    data['text_change_frequency'] = data.groupby('id')['text_change'].transform('sum')
    #data['text_change_ratio'] = data['text_change_frequency'] / data['total_characters'].replace(0, np.nan)
    data['text_change_frequency'] = pd.to_numeric(data['text_change_frequency'], errors='coerce')

# Convert 'total_characters' to numeric type (assuming it's currently a string)
    data['total_characters'] = pd.to_numeric(data['total_characters'], errors='coerce')

# Replace NaN values with 0 in 'total_characters' (or any other default value you prefer)
    data['total_characters'] = data['total_characters'].fillna(0)

# Avoid division by zero by replacing 0 in 'total_characters' with NaN
    data['total_characters'] = data['total_characters'].replace(0, np.nan)

# Perform division
    data['text_change_ratio'] = data['text_change_frequency'] / data['total_characters']

    data["activity_duration"]=data.groupby(["id","activity",])["action_time"].transform('sum')
    data["activity_transition_count"]=data.groupby("id")["word_count"].transform("nunique")
    
    data["cursor_postion"]=data["cursor_position"]/data["total_characters"].replace(0,np.nan)
    data["word_count_change_rate"]=data.groupby("id")["word_count"].transform("diff")/data["time_interval"]
    
    
    data["word_count_trend"]=data.groupby("id")["word_count"].transform("diff").rolling(window=3).mean()
    
    
    data["event_duration_diff"]=data.groupby("id")["action_time"].transform("diff")
    
    
    tqdm.pandas(desc="Processing 'id'")
    data_grouped=data.groupby('id').progress_apply(lambda x:x)
    
    final_data = data.groupby('id').agg({
        'event_rate': 'mean',
        'event_duration_variation': 'mean',
        'event_sequence_change': 'mean',
        'event_transition_count': 'last',
        'unique_events_count': 'last',
        'text_change_ratio': 'mean',
        'activity_duration': 'mean',
        'activity_transition_count': 'last',
        'cursor_position': 'mean',
        'word_count_change_rate': 'mean',
        'event_duration_diff': 'mean',
        #'score': 'last' 
    }).reset_index()

    return final_data

#tqdm.pandas(desc="Processing 'id'")
#data_grouped = data.groupby('id').progress_apply(lambda x: x)


path="/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv"
data=pd.read_csv(path)
features=feature_extraction(data)
directory_path = '/kaggle/temp'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
file_path=os.path.join(directory_path,'trimmed_features.csv')
features.to_csv(file_path, index=False)
print(f"Features saved to temporary file: {file_path}")
