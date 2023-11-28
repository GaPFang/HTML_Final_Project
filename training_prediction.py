# training
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

train_x = []
train_y = []
for key in tqdm(sbi_all):
    for i in range(7):
        for j in range(24*12):
            for k in range(len(sbi_all[key][i][j*5])):
                train_x.append([i, j*5, sbi_all[key][i][j*5][k], tot_all[key][i][j*5][k]])
                train_y.append(sbi_all[key][i][j*5][k])

train_x = np.array(train_x)
train_y = np.array(train_y)
model = LinearRegression()
model.fit(train_x, train_y)

# predict
predict_days = [(datetime(2023, 10, 21) + timedelta(days=i)).strftime('%Y%m%d') for i in range(4)] + [(datetime(2023, 12, 4) + timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]

with open('submission.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'sbi'])
    for date in tqdm(predict_days):
        day = (datetime.strptime(date, '%Y%m%d')).weekday()
        with open('./html.2023.final.data/sno_test_set.txt', 'r') as file:
            for station in tqdm(file, leave=False):
                station = station.replace('\n', '')
                tot = tot_all[station][0][0][0]
                for t in range(72):
                    pred_x = []
                    time = (datetime.strptime(date, "%Y%m%d")+timedelta(minutes=t*20)).strftime('%H:%M')
                    if t == 0:
                        sbi_pre = np.mean(list(map(float, sbi_mean[station][day][0:11])))
                    else:
                        sbi_pre = np.mean(list(map(float, sbi_mean[station][day][t*20-10:t*20+11])))
                    pred_x.append([day, t*20, sbi_pre, tot])
                    pred_x = np.array(pred_x)
                    predict = float(model.predict(pred_x))
                    if predict < 0:
                        predict = 0
                    elif predict > tot:
                        predict = tot
                    id = date+'_'+station+'_'+time
                    writer.writerow([id, predict])
