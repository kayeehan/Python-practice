# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 12:18:38 2022

@author: Hankm
"""
#취소되는 경기장의 날씨와 미세먼지를 확인하여 취소될 확률을 예측함
import pickle
import re
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import time
from collections import Counter
from urllib.request import urlopen
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup 
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

#kbo의 야구 경기 리스트
driver = webdriver.Chrome('c:/data_bigdata/chromedriver.exe')
driver.get('https://www.koreabaseball.com/Schedule/Schedule.aspx')

df = DataFrame(columns=['일자','시간','경기장','취소여부'])

driver.find_element(By.CSS_SELECTOR,'select#ddlYear > option:nth-child(2)').click() #2021
driver.find_element(By.CSS_SELECTOR,'select#ddlMonth > option:nth-child(4)').click() #04
time.sleep(2)
soup = BeautifulSoup(driver.page_source)
year = soup.select_one('select#ddlYear> option:nth-child(2)').text
table = [i for i in soup.select('tbody > tr > td')]
date = []
for i in table:
    try:
        if i.attrs['class'] == ['day']:
            date_tmp = year+'.'+i.text
        elif i.attrs['class'] == ['time']:
            date.append(date_tmp)
    except:
        pass
base_time = [i.text for i in soup.select('tbody > tr > td.time')]
for i in soup.select('td.day'):
    i.extract()
stadium = [i.text for i in soup.select('tbody > tr > td:nth-child(7)')]
cancellation = [i.text for i in soup.select('tbody > tr > td:nth-child(8)')]
df = df.append(DataFrame({'일자':date,'시간':base_time,'경기장':stadium,'취소여부':cancellation}),ignore_index=True)
time.sleep(1)

for j in range(5,11):
    driver.find_element(By.CSS_SELECTOR,'select#ddlMonth > option:nth-child('+str(j)+')').click()
    year = soup.select_one('select#ddlYear> option:nth-child(2)').text
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source)
    table = [i for i in soup.select('tbody > tr > td')]
    date = []
    for i in table:
        try:
            if i.attrs['class'] == ['day']:
                date_tmp = year+'.'+i.text
            elif i.attrs['class'] == ['time']:
             date.append(date_tmp)
        except:
            pass
    base_time = [i.text for i in soup.select('tbody > tr > td.time')]
    for i in soup.select('td.day'):
        i.extract()
    stadium = [i.text for i in soup.select('tbody > tr > td:nth-child(7)')]
    cancellation = [i.text for i in soup.select('tbody > tr > td:nth-child(8)')]
    df = df.append(DataFrame({'일자':date,'시간':base_time,'경기장':stadium,'취소여부':cancellation}),ignore_index=True)


for k in range(3,8):
    driver.find_element(By.CSS_SELECTOR,'select#ddlYear > option:nth-child('+str(k)+')').click() 
    time.sleep(0.7)
    for j in range(3,11):
        driver.find_element(By.CSS_SELECTOR,'select#ddlMonth > option:nth-child('+str(j)+')').click()
        year = soup.select_one('select#ddlYear > option:nth-child('+str(k)+')').text
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source)
        table = [i for i in soup.select('tbody > tr > td')]
        date = []
        for i in table:
            try:
                if i.attrs['class'] == ['day']:
                    date_tmp = year+'.'+i.text
                elif i.attrs['class'] == ['time']:
                    date.append(date_tmp)
            except:
                pass
        base_time = [i.text for i in soup.select('tbody > tr > td.time')]
        for i in soup.select('td.day'):
            i.extract()
        stadium = [i.text for i in soup.select('tbody > tr > td:nth-child(7)')]
        cancellation = [i.text for i in soup.select('tbody > tr > td:nth-child(8)')]
        df = df.append(DataFrame({'일자':date,'시간':base_time,'경기장':stadium,'취소여부':cancellation}),ignore_index=True)

df.info()

date=[]
for i in df_new['일자'].str.split('('):
    date.append(i[0])

df['일자'] = pd.to_datetime(df['일자'],format = '%Y.%m.%d')
df['일자'] = df['일자'].dt.strftime('%Y-%m-%d')

with open('c:/data_bigdata/baseball_list.csv','wb') as file:
    pickle.dump(df,file)

with open('c:/data_bigdata/baseball_list.csv','rb') as file:
    df_new = pickle.load(file)


#지역의 일자별 강수량 추출
stn_lst = {108:'서울',112:'인천',119:'수원',155:'창원',159:'부산',143:'대구',156:'광주',
           133:'대전',138:'포항',152:'울산',131:'청주'}

driver = webdriver.Chrome('c:/data_bigdata/chromedriver.exe')
result = DataFrame(columns=['일자','value','지역'])
for k in stn_lst.keys():
    for i in range(2016,2022):
        driver.get('https://www.weather.go.kr/w/obs-climate/land/past-obs/obs-by-element.do?stn='+str(k)+'&yy='+str(i)+'&obs=21')
        time.sleep(1)
        soup = BeautifulSoup(driver.page_source)
        month = [j.text for j in soup.select('tr.tablesorter-headerRow > th')]
        month.remove('일자')
        rain = [j.text.strip() for j in soup.select('#weather_table>tbody>tr>td')]
        day = [rain[j] for j in range(0,416,13)]
        for j in day:
            rain.remove(j)
        baseball_pivot = DataFrame(np.array(rain).reshape(32,12),columns=month,index=day)
        baseball_pivot = baseball_pivot.drop('합계')
        baseball_pivot = baseball_pivot.reset_index()
        baseball_pivot = baseball_pivot.replace('', np.nan).fillna(0)
        id_var = list(baseball_pivot.columns)
        baseball_pivot.iloc[:,1:12].astype('float')
        baseball_unpivot = pd.melt(baseball_pivot,id_vars=['index'],value_vars=id_var)
        baseball_unpivot.insert(0,'일자',str(i)+'년'+baseball_unpivot['variable']+baseball_unpivot['index'])
        baseball_unpivot = baseball_unpivot.drop(['index','variable'],axis=1)
        for j in soup.select('option'): 
            if j.attrs.get('value') == str(k):
                baseball_unpivot['지역'] = j.text.split('(')[0]
        result = result.append(baseball_unpivot,ignore_index=True)
        time.sleep(2)  
result
#일자가 2016년1월1일 형식이라 이를 df와 동일한 형식으로 변경해줌.
month=[]
day=[]
for i in result['일자'].str.split('년'):
    month.append(i[1].split('월')[0])
    day.append(i[1].split('월')[1][:-1])

month_zfill = []
for i in month:
    month_zfill.append(i.zfill(2))
day_zfill = []
for i in day:
    day_zfill.append(i.zfill(2))

result['일자'] = result['일자'].str[0:4]+'-'+month_zfill+'-'+day_zfill


with open('c:/data_bigdata/rainweather_list.csv','wb') as file:
    pickle.dump(result,file)
    
with open('c:/data_bigdata/rainweather_list.csv','rb') as file:
    rain = pickle.load(file)

#고척은 실내 경기장이기에 삭제함
df_new = df_new.loc[df_new['경기장'] != '고척',]
df_new = df_new.reset_index(drop=True)
#마산은 창원으로 지역명 변경되어 수정함
df_new.loc[df_new['경기장']=='마산','경기장'] = '창원'
#경기장명과 지역명 통일
city_name = {'서울':'잠실', '인천':'문학', '수원':'수원', '창원':'창원', '부산':'사직', 
 '대구':'대구', '광주':'광주', '대전':'대전', '포항':'포항', '울산':'울산', '청주':'청주'}

stadium = []
for i in rain['지역']:
    stadium.append(city_name[i])

rain['경기장'] = stadium

#경기장 우천취소 항목과 지역별 강수량 merge
df_new.일자 = df_new.일자.astype('str')
mg_rain = pd.merge(df_new,rain,how='left',on=['일자','경기장'])[['일자','경기장','취소여부','value']]

mg_rain.loc[mg_rain['취소여부']=='-','취소여부'] = '정상 경기 진행'
#우천취소와 정상경기 진행만 출력
mg_rain_cancel = mg_rain.loc[mg_rain['취소여부'].str.contains('우천취소')| mg_rain['취소여부'].str.contains('정상 경기 진행'),]
mg_rain_cancel['value'] = mg_rain_cancel['value'].astype('float64')

#시각화를 위해 범주형을 수치형으로 변경
mg_rain_cancel['취소여부'] = mg_rain_cancel.취소여부.map({'우천취소':0,'정상 경기 진행':1})
sns.stripplot(x=mg_rain_cancel['취소여부'],y=mg_rain_cancel.value)
mg_rain_cancel['취소여부'] = mg_rain_cancel.취소여부.map({0:'우천취소',1:'정상 경기 진행'})
sns.boxplot(y=mg_rain_cancel.value,x=mg_rain_cancel.취소여부)
sns.violinplot(y=mg_rain_cancel.value,x=mg_rain_cancel.취소여부)


#우천취소 및 정상 경기 진행에 따른 강수량 여부
mg_rain_cancel['취소여부'].unique()
mg_rain_cancel.loc[mg_rain_cancel['취소여부']=='우천취소','value'].describe()
mg_rain_cancel.loc[(mg_rain_cancel['취소여부']=='우천취소')& (mg_rain_cancel['value']==0),]
mg_rain_cancel.loc[(mg_rain_cancel['취소여부']=='정상 경기 진행')& (mg_rain_cancel['value']>=100),]


#강수량과 취소여부 학습데이터 및 테스트 데이터로 분할 후 학습
x_train,x_test,y_train,y_test  = train_test_split(mg_rain_cancel['value'],mg_rain_cancel['취소여부'],test_size=0.2)
Counter(y_train)
Counter(y_test)

nb = MultinomialNB()
nb.fit((np.array(x_train)).reshape(-1,1),y_train)

y_predict = nb.predict((np.array(x_test)).reshape(-1,1))
accuracy_score(y_test,y_predict) #0.9196
print(classification_report(y_test,y_predict))

confusion_matrix(y_test, y_predict)
'''array([[  0,  68],
           [  0, 778]]
'''

#knn방법으로 테스트
train = []
test = []
for k in range(1,42,2):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit((np.array(x_train)).reshape(-1,1),y_train)
    train.append(clf.score((np.array(x_train)).reshape(-1,1),y_train))
    test.append(clf.score((np.array(x_test)).reshape(-1,1),y_test))
print(classification_report(y_test,clf.predict(np.array(x_test).reshape(-1,1))))

clf = KNeighborsClassifier(n_neighbors=9)
clf.fit((np.array(x_train)).reshape(-1,1),y_train)
clf.score((np.array(x_test)).reshape(-1,1),y_test)

confusion_matrix(y_test,clf.predict(np.array(x_test).reshape(-1,1)))

plt.figure(figsize=(8,5))
plt.plot(range(1,42,2),train,label='train')
plt.plot(range(1,42,2),test,label='test')
plt.legend() #최적 n = 9























































