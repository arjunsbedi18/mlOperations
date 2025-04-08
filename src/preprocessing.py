import pandas as np

kobe = pd.read_csv('/Users/arjunbedi/Documents/gradSchool/mod4/mlOps/mlOperations/labs/lab2/kobe-bryant-shot-selection/data.csv.zip')

def encode_column(col):
    col_encode = {}
    i = 1
    for c in col:
        col_encode[c] = i
        i += 1
    return col_encode

actions_encode = encode_column(actions.keys())
seasons_encode = encode_column(seasons.keys())
shot_zones_encode = encode_column(shot_zones.keys())
df['action_type'] = df['action_type'].map(actions_encode)
df['season'] = df['season'].map(seasons_encode)
df['shot_zone_basic'] = df['shot_zone_basic'].map(shot_zones_encode)
