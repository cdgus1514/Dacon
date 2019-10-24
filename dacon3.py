import pandas as pd # 데이터 전처리
import numpy as np # 데이터 전처리
import matplotlib.pyplot as plt # 데이터 시각화
import itertools
from datetime import datetime, timedelta # 시간 데이터 처리

from fbprophet import Prophet
import logging

# from keras.models import Sequential
# from keras.models import Dense, LSTM

# prophet 로그 끄기
logging.getLogger('fbprophet').setLevel(logging.WARNING)



test = pd.read_csv("test2.csv")
# submission = pd.read_csv("submission_1002.csv")
# submission = pd.read_csv("submission_test.csv") # 24h 인덱스만
submission = pd.read_csv("submission_test2.csv") # month 인덱스만

# Time을 datetime의 형태로 변환
test['Time'] = pd.to_datetime(test['Time']) 
# test['floor'] = 0.000000000
test = test.set_index('Time')
# print(test.head)
# print(test.describe)


##  패널 데이터의 형태로 정리

# 빈 리스트를 생성합니다.
place_id=[]
time=[]
target=[]
for i in test.columns:
    for j in range(len(test)):
        place_id.append(i) # place_id에 미터 ID를 정리합니다.
        time.append(test.index[j]) # time에 시간대를 정리합니다.
        target.append(test[i].iloc[j]) # target에 전력량을 정리합니다.

new_df=pd.DataFrame({'place_id':place_id, 'time':time, 'target':target})
new_df=new_df.dropna() # 결측치를 제거합니다.
new_df=new_df.set_index('time') # time을 인덱스로 저장합니다.




## prophet 모델링

count = 0
agg={}
for key in new_df['place_id'].unique(): # 미터ID 200개의 리스트를 unique()함수를 통해 추출합니다.
    temp = new_df.loc[new_df['place_id']==key] # 미터ID 하나를 할당합니다.
    temp_1h=temp.resample('1h').sum() # 1시간 단위로 정리합니다.
    temp_1d=temp.resample('D').sum() # 1일 단위로 정리합니다.


# print("\n\n[DEBUG1-1]temp\n", temp, end="\n\n")
# print("\n\n[DEBUG1-2]temp_1h\n", temp_1h, end="\n\n")
# print("\n\n[DEBUG1-3]temp_1d\n", temp_1d, end="\n\n")

    
    temp_1h['ds'] = temp_1h.index
    temp_1h = pd.DataFrame(temp_1h, columns=["ds","target"])        # 순서변경
    temp_1h = temp_1h.rename(columns={"ds":"ds", "target":"y"})     # 이름 변경
    # print("\n\n[DEBUG1-2]process temp_1h\n", temp_1h, end="\n\n")

    temp_1d['ds'] = temp_1d.index
    temp_1d = pd.DataFrame(temp_1d, columns=["ds", "target"])
    temp_1d = temp_1d.rename(columns={"ds":"ds", "target":"y"})
    # print("\n\n[DEBUG1-3]process temp_1d\n", temp_1d, end="\n\n")


    '''
    # if count > 3:
    #     break


    # 시간별 예측
    model_24 = Prophet()
    model_24.fit(temp_1h)

    future_24 = model_24.make_future_dataframe(periods=24, freq="H")  # 24시간치 예측
    forecast_24 = model_24.predict(future_24)
    print("\n\n[yhat(24h)]\n",forecast_24[['ds', 'yhat']].tail(24), end="\n\n")

    test_24 = forecast_24.yhat.tolist()

    cntEnd = len(test_24)
    cntStart = cntEnd-24
    # print("\n\n[test list check]\n", test[cntStart:cntEnd], end="\n\n")
    # print(test_24[cntEnd-1])


    a = pd.DataFrame()

    for i in range(24):
        a['X2018_7_1_'+str(i+1)+'h'] = [test_24[cntStart+i]]



    # 일별 예측
    model_1d = Prophet()
    model_1d.fit(temp_1d)

    future_1d = model_1d.make_future_dataframe(periods=10)  # 10일치 예측
    forecast_1d = model_1d.predict(future_1d)
    print("\n\n[yhat(1d)]\n", forecast_1d[['ds', 'yhat']].tail(10), end="\n\n")

    test_1d = forecast_1d.yhat.tolist()

    cntEnd = len(test_1d)
    cntStart = cntEnd - 10
    # print("[test list check]\n", test_1d[cntStart:cntEnd], end="\n\n")

    for i in range(10):
        a['X2018_7_'+str(i+1)+'_d'] = [test_1d[cntStart+i]]
    '''
    
    
    # 월별 예측
    model_1m = Prophet()
    model_1m.fit(temp_1d)

    future_1m = model_1m.make_future_dataframe(periods=153)
    forecast_1m = model_1m.predict(future_1m)
    # print("\n\n[yhat(1m)]\n", forecast_1m[['ds', 'yhat']].tail(5), end="\n\n")

    test_1m = forecast_1m.yhat.tolist()
    a = pd.DataFrame()


    print("\n[test_1m check] (7m)\n", [np.sum(test_1m[0:31])], end="\n")
    print("\n[test_1m check] (8m)\n", [np.sum(test_1m[31:62])], end="\n")
    print("\n[test_1m check] (9m)\n", [np.sum(test_1m[62:92])], end="\n")
    print("\n[test_1m check] (10m)\n", [np.sum(test_1m[92:123])], end="\n")
    print("\n[test_1m check] (11m)\n", [np.sum(test_1m[123:153])], end="\n\n")


    a['X2018_7_m'] = [np.sum(test_1m[0:31])]
    a['X2018_8_m'] = [np.sum(test_1m[31:62])]
    a['X2018_9_m'] = [np.sum(test_1m[62:92])]
    a['X2018_10_m'] = [np.sum(test_1m[92:123])]
    a['X2018_11_m'] = [np.sum(test_1m[123:153])]


    

    count += 1
    a['meter_id'] = key 
    agg[key] = a[submission.columns.tolist()]
    print("\n\n" + str(count) + " >> " + key, end="\n\n")
# a = pd.DataFrame()
# pd.DataFrame(a).to_csv("./testprophet.csv", index=False)


# output
output1 = pd.concat(agg, ignore_index=False)
output2 = output1.reset_index().drop(['level_0','level_1'], axis=1)
output2['id'] = output2['meter_id'].str.replace('X','').astype(int)
output2 =  output2.sort_values(by='id', ascending=True).drop(['id'], axis=1).reset_index(drop=True)
output2.to_csv('sub_baseline_month.csv', index=False)
