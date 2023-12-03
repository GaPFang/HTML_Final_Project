import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

weather_df = pd.read_csv('weather/visualcrossing/merged_20231001_20231202_20231211.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df.drop(['stations', 'icon', 'name', 'conditions'], axis=1, inplace=True)
weather_df['preciptype'].replace({'rain': 1, np.nan: 0}, inplace=True)
helper = pd.DataFrame({'datetime': pd.date_range(start=weather_df['datetime'].min(), end=weather_df['datetime'].max(),freq='20T')})
weather_df = pd.merge(weather_df, helper, on='datetime', how='outer').sort_values('datetime')
weather_df.set_index('datetime', inplace=True)
weather_df = weather_df.interpolate(method='linear')
# print(weather_df)

ntu_stations = []
with open('sno_test_set.txt', 'r') as file:
    for line in file:
        # Remove any whitespace characters (like newline) and append to the list
        ntu_stations.append(line.strip())

# Load station bike data
station_20_dfs = {}
integrated_df = {}
for station in tqdm(ntu_stations):
    station_20_dfs[station] = pd.read_csv('20_min_data/20_'+station+'.csv', index_col=0)
    station_20_dfs[station].index = pd.to_datetime(station_20_dfs[station].index)
    filtered_station_data = station_20_dfs[station][(station_20_dfs[station].index >= '2023-10-02') & (station_20_dfs[station].index < '2023-12-3')][['sbi', 'total_min', 'day_of_week']]
    filtered_weather_data = weather_df[(weather_df.index >= '2023-10-2') & (weather_df.index < '2023-12-3')][['preciptype']]
    integrated_df[station] = pd.merge(
        filtered_station_data, filtered_weather_data, left_index=True, right_index=True)
    integrated_df[station].reset_index(inplace=True, names='datetime')
# print(integrated_df['500101001'])
# print(integrated_df['500101001'][(integrated_df['500101001']['datetime'] >= '20231021') & (integrated_df['500101001']['datetime'] <= '20231025')])

completed_df = {station: data.dropna() for station, data in integrated_df.items()}
# print(completed_df['500101001'])
# print(completed_df['500101001'][(completed_df['500101001']['datetime'] >= '20231021') & (completed_df['500101001']['datetime'] <= '20231026')])

y = {station: data['sbi'] for station, data in completed_df.items()}
X = {station: data[['datetime', 'total_min', 'day_of_week', 'preciptype']] for station, data in completed_df.items()}

predict_days = pd.date_range(start='2023-10-21', end='2023-10-24').tolist() + pd.date_range(start='2023-12-04', end='2023-12-10').tolist()

pred_time_slots = []
for date in predict_days:
    time_range = pd.date_range(start=date.replace(hour=0, minute=0, second=0),
                               end=date.replace(hour=23, minute=59, second=59),
                               freq='20T')
    pred_time_slots.extend(time_range)
pred_X = pd.DataFrame({'datetime': pred_time_slots})
pred_X['total_min'] = pred_X['datetime'].dt.hour*60 + pred_X['datetime'].dt.minute
pred_X['day_of_week'] = pred_X['datetime'].dt.dayofweek
pred_X['preciptype'] = pred_X['datetime'].apply(lambda x: weather_df.loc[x]['preciptype'])
# print(pred_X)

y_pred = {}
for station in tqdm(ntu_stations):
    X_train = X[station][['total_min', 'day_of_week', 'preciptype']]
    regressor = RandomForestRegressor()
    regressor.fit(X_train, y[station])
    y_pred[station] = regressor.predict(pred_X[['total_min', 'day_of_week', 'preciptype']])

pred_df = pd.DataFrame(columns=['datetime', 'sbi'])
for station in tqdm(ntu_stations):
    merged_df = pd.concat([pred_X['datetime'], pd.DataFrame({'sbi': y_pred[station]})], axis=1)
    merged_df['datetime'] = merged_df['datetime'].dt.strftime(f'%Y%m%d_{station}_%H:%M:%S')
    pred_df = pd.concat([pred_df, merged_df], axis=0)
pred_df.rename(columns={'datetime': 'id'})
print(pred_df)
merged_df.to_csv('random_forest_prediction.csv', index=False, header=['id', 'sbi'])
    

# station_score = {}
# for station in tqdm(ntu_stations):
#     X_train, X_test, y_train, y_test = train_test_split(X[station], y[station], shuffle=False)
#     split_data[station] = [X_train, X_test, y_train, y_test]
#     X_train = X_train[['total_min', 'day_of_week', 'preciptype']]
#     X_test = X_test[['total_min', 'day_of_week', 'preciptype']]
#     regressor = RandomForestRegressor()
#     regressor.fit(X_train, y_train)
#     y_pred[station] = regressor.predict(X_test)
#     station_score[station] = regressor.score(X_test, y_test)
#     break
# 
# for station in ntu_stations:
#     print(f'{station}: {station_score[station]}')
#     break
# 
# # print(split_data[station][1])
# # print(y_pred['500101001'])
# 
# # visualising
# plt.plot(split_data['500101001'][1]['datetime'], split_data['500101001'][3], color='blue')
# plt.plot(split_data['500101001'][1]['datetime'], y_pred[station], color='red')
# plt.savefig('testdata.png', dpi=300)