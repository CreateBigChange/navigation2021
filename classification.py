import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.utils.data as Data
import torch
from torch import nn


def label_wifi(base_path,WIFI2,mac_address_2,ref1):
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

        if time0 == item[0]:

            wifi2_[num, idx[0, 0] + 1] = RSSI

        else:
            num = num + 1
            time0 = item[0]
            wifi2_[num, 0] = time0
            wifi2_[num, idx[0, 0] + 1] = RSSI

    wifi2 = wifi2_[0:num + 1, :]

    del wifi2_

    sign = -1
    num = 0
    wifi2_label = np.zeros((ref1.shape[0], wifi2.shape[1]))
    width = wifi2.shape[1]
    for i in range(wifi2.shape[0]):
        idx = round(wifi2[i, 0] / 0.5)
        if idx >= wifi2_label.shape[0]:
            break;
        if sign != idx:
            if sign == -1:  # 判断是不是初始
                sign = idx
                num = 1
                wifi2_label[sign, 0:width - 1] = wifi2[i, 1:]

                wifi2_label[sign, width - 1] = sign
            else:

                # old

                wifi2_label[sign, 0:width - 1] = wifi2_label[sign, 0:width - 1] / num

                # new
                sign = idx
                wifi2_label[sign, 0:width - 1] = wifi2[i, 1:]

                wifi2_label[sign, width - 1] = sign
                num = 1

        else:
            num = num + 1
            wifi2_label[sign, 0:width - 1] = wifi2_label[sign, 0:width - 1] + wifi2[i, 1:]

    wifi2_label = wifi2_label[0:sign + 1, :]
    return wifi2_label


def label_ios_mag(base_path,IOS_ACCLE,IOS_BLE,IOS_GYRO,IOS_MAG,ref1):
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

    sign = -1
    num = 0
    ios_mag_label = np.zeros((ref1.shape[0], ios_mag.shape[1]))
    print(base_path)
    for i in range(ios_mag.shape[0]):
        idx = round(ios_mag.iloc[i, 0] / 0.5)
        if idx >= ios_mag_label.shape[0]:
            break;

        if sign != idx:
            if sign == -1:  # 判断是不是初始
                sign = idx
                num = 1

                ios_mag_label[sign, 0] = ios_mag.iloc[i, 1]
                ios_mag_label[sign, 1] = ios_mag.iloc[i, 2]
                ios_mag_label[sign, 2] = ios_mag.iloc[i, 3]
                ios_mag_label[sign, 3] = sign
            else:

                # old
                ios_mag_label[sign, 0] = ios_mag_label[sign, 0] / num
                ios_mag_label[sign, 1] = ios_mag_label[sign, 1] / num
                ios_mag_label[sign, 2] = ios_mag_label[sign, 2] / num

                # new
                sign = idx
                print(sign)
                print(ios_mag_label.shape)
                print(ios_mag.shape)
                ios_mag_label[sign, 0] = ios_mag.iloc[i, 1]
                ios_mag_label[sign, 1] = ios_mag.iloc[i, 2]
                ios_mag_label[sign, 2] = ios_mag.iloc[i, 3]
                ios_mag_label[sign, 3] = sign
                num = 1

        else:
            num = num + 1
            ios_mag_label[idx, 0] = ios_mag_label[idx, 0] + ios_mag.iloc[i, 1]
            ios_mag_label[idx, 1] = ios_mag_label[idx, 1] + ios_mag.iloc[i, 2]
            ios_mag_label[idx, 2] = ios_mag_label[idx, 2] + ios_mag.iloc[i, 3]

    ios_mag_label = ios_mag_label[0:sign + 1, :]
    return ios_mag_label

def label_sensor2(base_path,SENSOR2,ref1):
    sensor2 = pd.read_csv(os.path.join(base_path, SENSOR2))
    sensor2.columns = ["time", "x_gyro", "y_gyro", "z_gyro", "x_accel", "y_accel", "z_accel", "x_mag", "y_mag",
                       "z_mag",
                       "baro", "temper"]

    sign = -1
    num = 0
    sensor2_label = np.zeros((ref1.shape[0], sensor2.shape[1]))

    for i in range(sensor2.shape[0]):

        idx = round(sensor2.iloc[i, 0] / 0.5)
        if idx >= sensor2_label.shape[0]:
            break;
        if sign != idx:
            if sign == -1:  # 判断是不是初始
                sign = idx
                num = 1

                sensor2_label[sign, 0] = sensor2.iloc[i, 1]
                sensor2_label[sign, 1] = sensor2.iloc[i, 2]
                sensor2_label[sign, 2] = sensor2.iloc[i, 3]
                sensor2_label[sign, 3] = sensor2.iloc[i, 4]
                sensor2_label[sign, 4] = sensor2.iloc[i, 5]
                sensor2_label[sign, 5] = sensor2.iloc[i, 6]
                sensor2_label[sign, 6] = sensor2.iloc[i, 7]
                sensor2_label[sign, 7] = sensor2.iloc[i, 8]
                sensor2_label[sign, 8] = sensor2.iloc[i, 9]
                sensor2_label[sign, 9] = sensor2.iloc[i, 10]
                sensor2_label[sign, 10] = sensor2.iloc[i, 11]
                sensor2_label[sign, 11] = sign

            else:

                # old
                sensor2_label[sign, 0] = sensor2_label[sign, 0] / num
                sensor2_label[sign, 1] = sensor2_label[sign, 1] / num
                sensor2_label[sign, 2] = sensor2_label[sign, 2] / num
                sensor2_label[sign, 3] = sensor2_label[sign, 3] / num
                sensor2_label[sign, 4] = sensor2_label[sign, 4] / num
                sensor2_label[sign, 5] = sensor2_label[sign, 5] / num
                sensor2_label[sign, 6] = sensor2_label[sign, 6] / num
                sensor2_label[sign, 7] = sensor2_label[sign, 7] / num
                sensor2_label[sign, 8] = sensor2_label[sign, 8] / num
                sensor2_label[sign, 9] = sensor2_label[sign, 9] / num
                sensor2_label[sign, 10] = sensor2_label[sign, 10] / num

                # new
                sign = idx
                sensor2_label[sign, 0] = sensor2.iloc[i, 1]
                sensor2_label[sign, 1] = sensor2.iloc[i, 2]
                sensor2_label[sign, 2] = sensor2.iloc[i, 3]
                sensor2_label[sign, 3] = sensor2.iloc[i, 4]
                sensor2_label[sign, 4] = sensor2.iloc[i, 5]
                sensor2_label[sign, 5] = sensor2.iloc[i, 6]
                sensor2_label[sign, 6] = sensor2.iloc[i, 7]
                sensor2_label[sign, 7] = sensor2.iloc[i, 8]
                sensor2_label[sign, 8] = sensor2.iloc[i, 9]
                sensor2_label[sign, 9] = sensor2.iloc[i, 10]
                sensor2_label[sign, 10] = sensor2.iloc[i, 11]
                sensor2_label[sign, 11] = sign
                num = 1

        else:

            num = num + 1
            sensor2_label[idx, 0] = sensor2_label[idx, 0] + sensor2.iloc[i, 1]
            sensor2_label[idx, 1] = sensor2_label[idx, 1] + sensor2.iloc[i, 2]
            sensor2_label[idx, 2] = sensor2_label[idx, 2] + sensor2.iloc[i, 3]
            sensor2_label[idx, 3] = sensor2_label[idx, 3] + sensor2.iloc[i, 4]
            sensor2_label[idx, 4] = sensor2_label[idx, 4] + sensor2.iloc[i, 5]
            sensor2_label[idx, 5] = sensor2_label[idx, 5] + sensor2.iloc[i, 6]
            sensor2_label[idx, 6] = sensor2_label[idx, 6] + sensor2.iloc[i, 7]
            sensor2_label[idx, 7] = sensor2_label[idx, 7] + sensor2.iloc[i, 8]
            sensor2_label[idx, 8] = sensor2_label[idx, 8] + sensor2.iloc[i, 9]
            sensor2_label[idx, 9] = sensor2_label[idx, 9] + sensor2.iloc[i, 10]
            sensor2_label[idx, 10] = sensor2_label[idx, 10] + sensor2.iloc[i, 11]
    sensor2_label = sensor2_label[0:sign + 1, :]
    return sensor2_label

def label_sensor3(base_path, SENSOR3, ref1):
    sensor3 = pd.read_csv(os.path.join(base_path, SENSOR3))
    sensor3.columns = ["time", "x_gyro", "y_gyro", "z_gyro", "x_accel", "y_accel", "z_accel", "x_mag", "y_mag",
                       "z_mag",
                       "baro", "temper"]

    sign = -1
    num = 0
    sensor3_label = np.zeros((ref1.shape[0], sensor3.shape[1]))

    for i in range(sensor3.shape[0]):

        idx = round(sensor3.iloc[i, 0] / 0.5)
        if idx >= sensor3_label.shape[0]:
            break;
        if sign != idx:
            if sign == -1:  # 判断是不是初始
                sign = idx
                num = 1

                sensor3_label[sign, 0] = sensor3.iloc[i, 1]
                sensor3_label[sign, 1] = sensor3.iloc[i, 2]
                sensor3_label[sign, 2] = sensor3.iloc[i, 3]
                sensor3_label[sign, 3] = sensor3.iloc[i, 4]
                sensor3_label[sign, 4] = sensor3.iloc[i, 5]
                sensor3_label[sign, 5] = sensor3.iloc[i, 6]
                sensor3_label[sign, 6] = sensor3.iloc[i, 7]
                sensor3_label[sign, 7] = sensor3.iloc[i, 8]
                sensor3_label[sign, 8] = sensor3.iloc[i, 9]
                sensor3_label[sign, 9] = sensor3.iloc[i, 10]
                sensor3_label[sign, 10] = sensor3.iloc[i, 11]
                sensor3_label[sign, 11] = sign

            else:

                # old
                sensor3_label[sign, 0] = sensor3_label[sign, 0] / num
                sensor3_label[sign, 1] = sensor3_label[sign, 1] / num
                sensor3_label[sign, 2] = sensor3_label[sign, 2] / num
                sensor3_label[sign, 3] = sensor3_label[sign, 3] / num
                sensor3_label[sign, 4] = sensor3_label[sign, 4] / num
                sensor3_label[sign, 5] = sensor3_label[sign, 5] / num
                sensor3_label[sign, 6] = sensor3_label[sign, 6] / num
                sensor3_label[sign, 7] = sensor3_label[sign, 7] / num
                sensor3_label[sign, 8] = sensor3_label[sign, 8] / num
                sensor3_label[sign, 9] = sensor3_label[sign, 9] / num
                sensor3_label[sign, 10] = sensor3_label[sign, 10] / num

                # new
                sign = idx
                sensor3_label[sign, 0] = sensor3.iloc[i, 1]
                sensor3_label[sign, 1] = sensor3.iloc[i, 2]
                sensor3_label[sign, 2] = sensor3.iloc[i, 3]
                sensor3_label[sign, 3] = sensor3.iloc[i, 4]
                sensor3_label[sign, 4] = sensor3.iloc[i, 5]
                sensor3_label[sign, 5] = sensor3.iloc[i, 6]
                sensor3_label[sign, 6] = sensor3.iloc[i, 7]
                sensor3_label[sign, 7] = sensor3.iloc[i, 8]
                sensor3_label[sign, 8] = sensor3.iloc[i, 9]
                sensor3_label[sign, 9] = sensor3.iloc[i, 10]
                sensor3_label[sign, 10] = sensor3.iloc[i, 11]
                sensor3_label[sign, 11] = sign
                num = 1

        else:

            num = num + 1
            sensor3_label[idx, 0] = sensor3_label[idx, 0] + sensor3.iloc[i, 1]
            sensor3_label[idx, 1] = sensor3_label[idx, 1] + sensor3.iloc[i, 2]
            sensor3_label[idx, 2] = sensor3_label[idx, 2] + sensor3.iloc[i, 3]
            sensor3_label[idx, 3] = sensor3_label[idx, 3] + sensor3.iloc[i, 4]
            sensor3_label[idx, 4] = sensor3_label[idx, 4] + sensor3.iloc[i, 5]
            sensor3_label[idx, 5] = sensor3_label[idx, 5] + sensor3.iloc[i, 6]
            sensor3_label[idx, 6] = sensor3_label[idx, 6] + sensor3.iloc[i, 7]
            sensor3_label[idx, 7] = sensor3_label[idx, 7] + sensor3.iloc[i, 8]
            sensor3_label[idx, 8] = sensor3_label[idx, 8] + sensor3.iloc[i, 9]
            sensor3_label[idx, 9] = sensor3_label[idx, 9] + sensor3.iloc[i, 10]
            sensor3_label[idx, 10] = sensor3_label[idx, 10] + sensor3.iloc[i, 11]
    sensor3_label = sensor3_label[0:sign + 1, :]
    return sensor3_label

def label_data(base_path):

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




    # ref
    ref1 = pd.read_csv(os.path.join(base_path, REF1), delimiter=' ')
    ref1.columns = ["time", "lat", "lon", "velocity", "heading"]
    ref2 = pd.read_csv(os.path.join(base_path, REF2), delimiter=' ', header=None)
    ref2.columns = ["time", "lat", "lon", "velocity", "heading"]

    # path
    base = os.path.join(base_path, 'classificaiotn')
    if not os.path.exists(base):
        os.mkdir(base)

    mac2 = pd.read_csv("mac2.csv", header = None);
    mac_address_2=mac2.values;

    mac3 = pd.read_csv("mac3.csv",header = None);
    mac_address_3 = mac3.values;

    #################### os.path.exists(os.path.join(base_path, IOS_ACCLE)) #####################
    if os.path.exists(os.path.join(base_path, IOS_ACCLE)):
        ios_mag_label=label_ios_mag(base_path=base_path,IOS_ACCLE=IOS_ACCLE,IOS_BLE=IOS_BLE,IOS_GYRO=IOS_GYRO,IOS_MAG=IOS_MAG,ref1=ref1)
        np.savetxt(os.path.join(base, "ios_mag.csv"), ios_mag_label, fmt='%.16f', delimiter=',')



    if os.path.exists(os.path.join(base_path,SENSOR2)):
        sensor2_label=label_sensor2(base_path=base_path, SENSOR2=SENSOR2, ref1=ref1)
        np.savetxt(os.path.join(base, "sensor2.csv"), sensor2_label, fmt='%.16f', delimiter=',')





    if os.path.exists(os.path.join(base_path,SENSOR3)):
        sensor3_label = label_sensor3(base_path = base_path, SENSOR3 = SENSOR3, ref1 = ref1)
        np.savetxt(os.path.join(base, "sensor3.csv"), sensor3_label, fmt='%.16f', delimiter=',')



    if os.path.exists(os.path.join(base_path, WIFI2)):
        wifi2_label=label_wifi(base_path, WIFI2, mac_address_2, ref1)

        np.savetxt(os.path.join(base, "wifi2.csv"), wifi2_label, fmt='%.16f', delimiter=',')


    if os.path.exists(os.path.join(base_path, WIFI3)):


        wifi3_label=label_wifi(base_path, WIFI3, mac_address_3, ref1)

        np.savetxt(os.path.join(base, "wifi3.csv"), wifi3_label, fmt='%.16f', delimiter=',')




if __name__ == '__main__':
    # define the folder path
    file_path_base = 'D:\developmentTool\Code\DATA\SCI 10 DATA _v1_150906'

    folder_path_You = os.path.join(file_path_base, 'Trajectory_You')
    folder_path_David = os.path.join(file_path_base, 'Trajectory_David')

    tajectory_path_You = os.listdir(folder_path_You)
    tajectory_path_David = os.listdir(folder_path_David)



    #list folder
    for idx,path_name in enumerate(tajectory_path_You):
        if idx==0 :
            continue;


        #base_path = os.path.join(folder_path_You, "20150903120304")
        base_path = os.path.join(folder_path_You, tajectory_path_You[idx])
        label_data(base_path)

    # base_path = os.path.join(folder_path_You, tajectory_path_You[3])
    # label_data(base_path)