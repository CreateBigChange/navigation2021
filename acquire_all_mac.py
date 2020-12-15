# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.utils.data as Data
import torch
from torch import nn

from get_pos import get_pos


mac2=set()
mac3=set()
if __name__ == '__main__':

    # define name of files
    IOS_ACCLE = "ios_accel_1.csv"
    IOS_BLE = "ios_ble_1.csv"
    IOS_GYRO = "ios_gyro_1.csv"
    IOS_MAG = "ios_mag_1.csv"
    REF1 = "ref_1.txt"
    REF2 = "ref_2.txt"
    SENSOR2 = "sensors_2_matrix.txt"
    SENSOR3 = "sensors_3_matrix.txt"
    LANDMARK = "landmark.txt"
    WIFI2 = "wifi_2.txt"
    WIFI3 = "wifi_3.txt"

    #define the folder path
    file_path_base = 'D:\developmentTool\Code\DATA\SCI 10 DATA _v1_150906'

    folder_path_You=os.path.join(file_path_base,'Trajectory_You')
    folder_path_David=os.path.join(file_path_base,'Trajectory_David')

    tajectory_path_You=os.listdir(folder_path_You)
    tajectory_path_David=os.listdir(folder_path_David)


    ####################read file########################


    #list folder
    wifi2_all=pd.DataFrame([])
    wifi3_all=pd.DataFrame([])

    for idx,path_name in enumerate(tajectory_path_You):
        if path_name=='.DS_Store' :
            continue;

        base_path = os.path.join(folder_path_You, tajectory_path_You[idx])
        print(base_path)


        if os.path.exists(os.path.join(base_path, WIFI2)):
            wifi2 = pd.read_csv(os.path.join(base_path, WIFI2))
            # trans wifi2 to dataframe
            temp = []
            for idx, item in wifi2.iterrows():
                temp.append(item.str.split(' ')[0][:])

            wifi2 = pd.DataFrame(temp)
            mac_address_2 = np.array(list(set(wifi2.iloc[:, 1])))
            # aquired all mac address from wifi2
            mac2=mac2.union(mac_address_2)
            # mac_address_2 = np.array(list(set(wifi2.iloc[:, 1])))


        if os.path.exists(os.path.join(base_path, WIFI3)):
            wifi3 = pd.read_csv(os.path.join(base_path, WIFI3))

            # trans wifi3 to dataframe
            temp = []
            for idx, item in wifi3.iterrows():
                temp.append(item.str.split(' ')[0][:])
            wifi3 = pd.DataFrame(temp)


            mac_address_3 = np.array(list(set(wifi3.iloc[:, 1])))
            mac3=mac3.union(mac_address_3)

    print(1)