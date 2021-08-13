# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:05:49 2021

@author: Steve
"""
import random
from datetime import datetime
import time
import numpy as np
import pandas as pd
ts = time.time()

end = 1627481198.0125883
start = 1609510074.000

time_list, date_list = [], []
phone_det_list, vehicle_det_list = [], []
time_stamps = [random.random() * (end - start) + start for _ in range(1890)]

for dates, times in zip(time_stamps, time_stamps):
    date_list.append(datetime.fromtimestamp(dates).strftime('%d/%m/%y'))
    time_list.append(datetime.fromtimestamp(times).strftime('%H:%M:%S'))


#Set values for mean and standard deviation
phone_mean = 1.2
phone_std_dev = 0.3
vehicle_mean = 50
vehicle_std_dev = 10

for i in range(1890):
    phone_det_list.append(round(np.random.normal(phone_mean, phone_std_dev)))
    vehicle_det_list.append(round(np.random.normal(vehicle_mean, vehicle_std_dev)))


date_list = np.array(date_list)


df = pd.DataFrame()


df['Date'] = date_list
df['Time'] = time_list
df['Timestamp'] = time_stamps
df['Phone Detections'] = phone_det_list
df['Vehicle Detections'] = vehicle_det_list

df.to_csv('./detections/summary/tester2.csv', encoding='utf-8', index=False)
