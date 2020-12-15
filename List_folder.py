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

    mac2 = pd.read_csv("mac2.csv", header = None);
    mac_address_2=mac2.values;

    mac3 = pd.read_csv("mac3.csv",header = None);
    mac_address_3 = mac3.values;
    ####################read file########################


    #list folder
    for idx,path_name in enumerate(tajectory_path_You):
        if idx==0 :
            continue;

        base_path = os.path.join(folder_path_You, tajectory_path_You[idx])




        # ref
        ref1 = pd.read_csv(os.path.join(base_path, REF1), delimiter=' ')
        ref1.columns = ["time", "lat", "lon", "velocity", "heading"]
        ref2 = pd.read_csv(os.path.join(base_path, REF2), delimiter=' ', header=None)
        ref2.columns = ["time", "lat", "lon", "velocity", "heading"]
        #path
        base=os.path.join(base_path,'prediction')
        if not os.path.exists(base):
            os.mkdir(base)
        # ############# handle IOS data ################
        #
        # os.path.exists(os.path.join(base_path, IOS_ACCLE))
        if os.path.exists(os.path.join(base_path, IOS_ACCLE)):

            ios_accle = pd.read_csv(os.path.join(base_path, IOS_ACCLE))
            ios_ble = pd.read_csv(os.path.join(base_path, IOS_BLE))
            ios_gyro = pd.read_csv(os.path.join(base_path, IOS_GYRO))
            ios_mag = pd.read_csv(os.path.join(base_path, IOS_MAG))
            ios_time0 = min(ios_accle.min().Time, ios_ble.min().Time, ios_gyro.min().Time, ios_mag.min().Time)

            # calibration Time
            ios_accle['Time'] = ios_accle['Time'].apply(lambda x: x - ios_time0)
            ios_ble['Time'] = ios_ble['Time'].apply(lambda x: x - ios_time0)
            ios_gyro['Time'] = ios_gyro['Time'].apply(lambda x: x - ios_time0)
            ios_mag['Time'] = ios_mag['Time'].apply(lambda x: x - ios_time0)

            # ios
            # ios_accle = get_pos(ios_accle, ref1)
            # ios_gyro = get_pos(ios_gyro, ref1)
            # ios_ble = get_pos(ios_ble, ref1)
            ios_mag = get_pos(ios_mag, ref1)

            # np.savetxt(os.path.join(base, "ios_accle.csv"), ios_accle.values, fmt='%.16f', delimiter=',')
            # np.savetxt(os.path.join(base, "ios_gyro.csv"), ios_gyro.values, fmt='%.16f', delimiter=',')
            # np.savetxt(os.path.join(base, "ios_ble.csv"), ios_ble.values, fmt='%.16f', delimiter=',')
            np.savetxt(os.path.join(base,"ios_mag.csv"),ios_mag.values,fmt='%.16f',delimiter=',')



        if os.path.exists(os.path.join(base_path,SENSOR2)):
            sensor2 = pd.read_csv(os.path.join(base_path, SENSOR2))
            sensor2.columns = ["time", "x_gyro", "y_gyro", "z_gyro", "x_accel", "y_accel", "z_accel", "x_mag", "y_mag",
                               "z_mag",
                               "baro", "temper"]
            sensor2 = get_pos(sensor2, ref1)
            np.savetxt(os.path.join(base, "sensor2.csv"),sensor2.values,fmt='%.16f',delimiter=',')



        if os.path.exists(os.path.join(base_path,SENSOR3)):
            sensor3 = pd.read_csv(os.path.join(base_path, SENSOR3))
            sensor3.columns = ["time", "x_gyro", "y_gyro", "z_gyro", "x_accel", "y_accel", "z_accel", "x_mag", "y_mag",
                               "z_mag",
                               "baro", "temper"]

            sensor3 = get_pos(sensor3, ref1)
            np.savetxt(os.path.join(base, "sensor3.csv"),sensor3.values,fmt='%.16f',delimiter=',')



        if os.path.exists(os.path.join(base_path,WIFI2)):
            wifi2 = pd.read_csv(os.path.join(base_path, WIFI2))
            # trans wifi2 to dataframe
            temp = []
            for idx, item in wifi2.iterrows():
                temp.append(item.str.split(' ')[0][:])
            wifi2 = pd.DataFrame(temp)


            # aquired all mac address from wifi2



            ######## reshape WIFI2 AND WIFI3 ##########
            wifi2_ = np.zeros((wifi2.shape[0], mac_address_2.shape[0] + 1))


            ### reshape WIFI2
            time0 = 0
            num = -1

            for i, item in wifi2.iterrows():

                mac = item[1]
                RSSI = item[2]
                idx = np.argwhere(mac == mac_address_2)
                print(i)
                if time0 == item[0]:

                    wifi2_[num, idx[0, 0] + 1] = RSSI

                else:
                    num = num + 1
                    time0 = item[0]
                    wifi2_[num, 0] = time0
                    wifi2_[num, idx[0, 0] + 1] = RSSI

            wifi2 = wifi2_[0:num + 1, :]

            del wifi2_

            wifi2 = get_pos(pd.DataFrame(wifi2), ref1)
            np.savetxt(os.path.join(base, "wifi2.csv"),wifi2.values,fmt='%.16f',delimiter=',')

        if os.path.exists(os.path.join(base_path, WIFI3)):
            wifi3 = pd.read_csv(os.path.join(base_path, WIFI3))

            # trans wifi3 to dataframe
            temp = []
            for idx, item in wifi3.iterrows():
                temp.append(item.str.split(' ')[0][:])
            wifi3 = pd.DataFrame(temp)


            wifi3_ = np.zeros((wifi3.shape[0], mac_address_3.shape[0] + 1))
            ### reshape WIFI3
            time0 = 0
            num = -1

            for i, item in wifi3.iterrows():

                mac = item[1]
                RSSI = item[2]
                idx = np.argwhere(mac == mac_address_3)

                if time0 == item[0]:

                    wifi3_[num, idx[0, 0] + 1] = RSSI

                else:
                    num = num + 1
                    time0 = item[0]
                    wifi3_[num, 0] = time0
                    wifi3_[num, idx[0, 0] + 1] = RSSI

            wifi3 = wifi3_[0:num + 1, :]
            del wifi3_

            wifi3 = get_pos(pd.DataFrame(wifi3), ref1)
            np.savetxt(os.path.join(base, "wifi3.csv"),wifi3.values,fmt='%.16f',delimiter=',')

        # sensor3\sensor2



        # others


        ##### reshape ble #####

        ### save result ###

        print(idx)
        print(path_name)


