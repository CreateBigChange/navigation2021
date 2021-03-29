"""
    ref_all's classification is from 0 to 1592 !!! important

"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch.utils.data as Data
import torch
from matplotlib import animation
from sklearn import preprocessing
from torch import nn
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from joblib import dump, load
import time

def read_data(persons,datatypes,test_idx,seperation_id,TIME_STEP,INPUT_SIZE):



    # reference points
    ref_names = ["ref_point_0.5m.csv", "ref_point_0.75m.csv", "ref_point_1.0m.csv", "ref_point_1.25m.csv",
                 "ref_point_1.5m.csv", "ref_point_1.75m.csv", "ref_point_2.0m.csv", "ref_point_2.5m.csv",
                 "ref_point_3.0m.csv", "ref_point_4.0m.csv"]
    # grid size for data
    names = ["seperation_0.5", "seperation_0.75", "seperation_1.0", "seperation_1.25", "seperation_1.5",
             "seperation_1.75", "seperation_2.0", "seperation_2.5", "seperation_3.0", "seperation_4.0"]

    # File of Data
    classification = "classification"


    David_Trajectorys = []
    You_Trajectorys = []

    David_files = os.listdir(os.path.join("./data/Trajectory_David"))
    You_files = os.listdir(os.path.join("./data/Trajectory_You"))

    for item in David_files:
        David_Trajectorys.append(os.path.join("./data/Trajectory_David", item))

    for item in You_files:
        You_Trajectorys.append(os.path.join("./data/Trajectory_You", item))







    all_refences = np.loadtxt("ref_point.csv", delimiter=',')
    # padding value  = min(RSSIs from train data and test data)
    padding = -99.0

    for path_idx in persons:
        if path_idx == 0:
            path = David_Trajectorys
            path = ['./data/Trajectory_David/20150903000101', './data/Trajectory_David/20150903000102',
                    './data/Trajectory_David/20150903000201', './data/Trajectory_David/20150903000202',
                    './data/Trajectory_David/20150903000301', './data/Trajectory_David/20150903000302',
                    './data/Trajectory_David/20150903000303', './data/Trajectory_David/20150903000304',
                    './data/Trajectory_David/20150903000305', './data/Trajectory_David/20150903000401',
                    './data/Trajectory_David/20150903000501', './data/Trajectory_David/20150903000601',
                    './data/Trajectory_David/20150903000602', './data/Trajectory_David/20150903001001',
                    './data/Trajectory_David/20150903001101', './data/Trajectory_David/20150903001201',
                    './data/Trajectory_David/20150903010101', './data/Trajectory_David/20150903010201',
                    './data/Trajectory_David/20150903010301', './data/Trajectory_David/20150903010401']
        else:

            path = You_Trajectorys
            path = ['./data/Trajectory_You/20150902115142', './data/Trajectory_You/20150903010338',
                    './data/Trajectory_You/20150903010718', './data/Trajectory_You/20150903010831',
                    './data/Trajectory_You/20150903011201', './data/Trajectory_You/20150903011715',
                    './data/Trajectory_You/20150903013053', './data/Trajectory_You/20150903013636',
                    './data/Trajectory_You/20150903014146', './data/Trajectory_You/20150903014459',
                    './data/Trajectory_You/20150903014735', './data/Trajectory_You/20150903020638',
                    './data/Trajectory_You/20150903021224', './data/Trajectory_You/20150903033517',
                    './data/Trajectory_You/20150903034855', './data/Trajectory_You/20150903035855',
                    './data/Trajectory_You/20150903040854', './data/Trajectory_You/20150903041905',
                    './data/Trajectory_You/20150903042430', './data/Trajectory_You/20150903043106',
                    './data/Trajectory_You/20150903120304', './data/Trajectory_You/20150903122250',
                    './data/Trajectory_You/20150903123218', './data/Trajectory_You/20150903124254']

        for datatype in datatypes:
            for i in range(len(path)):

                if i == test_idx:
                    continue

                if not os.path.exists(os.path.join(path[i], classification, names[seperation_id], datatype)):
                    continue
                if i == 5:
                    continue
                if i == 18:
                    continue

                train_data_ = np.loadtxt(os.path.join(path[i], classification, names[seperation_id], datatype), delimiter=',')

                macs = np.loadtxt("mac_WIFI.csv",dtype=str)

                corss_mac = np.loadtxt("cross_2017_origin.csv",dtype=str)

                idx_sort = []
                for _,item in enumerate(corss_mac):
                    idx = np.argwhere(item == macs)
                    idx_sort.append(idx[0, 0])

                train_data_temp = train_data_[:, idx_sort]
                train_data_ = np.hstack((train_data_[:, 0].reshape(-1, 1), train_data_temp, train_data_[:, -3:]))

                if not "train_data" in locals().keys():
                    # read data
                    train_data = train_data_.copy()
                    # padding
                    idx_zero = np.argwhere(train_data[:,0:-3] == 0)
                    train_data[idx_zero[:,0],idx_zero[:,1]] = padding

                    if train_data.shape[0] < TIME_STEP:
                        continue

                    # acquire Sequential
                    Sequential_train_data = Windows_Split_Module(train_data, TIME_STEP)

                    Sequential_train_datas = Sequential_train_data


                    Graident_train_data = Graident_Generator(Sequential_train_data)


                    Graident_train_datas = Graident_train_data

                else:

                    # padding
                    idx_zero = np.argwhere(train_data_[:, 0:-3] == 0)
                    train_data_[idx_zero[:, 0], idx_zero[:, 1]] =padding

                    if train_data_.shape[0] < TIME_STEP:
                        continue

                    Sequential_train_data = Windows_Split_Module(train_data_, TIME_STEP)
                    Sequential_train_datas = np.concatenate((Sequential_train_datas,Sequential_train_data), axis=0)

                    Graident_train_data = Graident_Generator(Sequential_train_data)
                    Graident_train_datas = np.concatenate((Graident_train_datas, Graident_train_data), axis=0)

                    train_data = np.vstack((train_data, train_data_))





    test_data = np.loadtxt(os.path.join(path[test_idx], classification, names[seperation_id], datatype), delimiter=',')
    test_data_2015 = np.loadtxt("data_2015_formatted.csv", delimiter=",")
    test_data_2017 = np.loadtxt("data_2017_formatted.csv", delimiter=",")
    test_data = test_data_2017
    # # padding
    idx_zero = np.argwhere(test_data[:, 0:-3] == 0)
    test_data[idx_zero[:, 0], idx_zero[:, 1]] = padding


    Sequential_test_data = Windows_Split_Module(test_data, TIME_STEP)
    Graident_test_data = Graident_Generator(Sequential_test_data)

    # train_data = np.loadtxt(os.path.join(path[0], classification, datatype), delimiter=',')
    #
    # test_data = np.loadtxt(os.path.join(path[1], classification, datatype), delimiter=',')


    ##################################### acquire train data and test data

    train_x = train_data[:, 1: - 3]
    train_y_c = train_data[:,  - 1].astype(int)
    train_y_r = train_data[:,  -3:- 1]

    test_x = test_data[:, 1: - 3]
    test_y_c = test_data[:, - 1].astype(int)
    test_y_r = test_data[:, - 3: - 1]



    all_refences = np.loadtxt(ref_names[seperation_id], delimiter=',')
    ################       lanmark all points with rssi
    refrence_rssi = np.zeros((all_refences.shape[0], INPUT_SIZE+3))
    for i, item in enumerate(all_refences):
        # print(item)
        loc_idx = item[-1]
        location = item[0:2]
        refrence_rssi[i, -3:-1] = location
        refrence_rssi[i, -1] = loc_idx
        # find all  fingerprints where location = item
        idx = np.argwhere(loc_idx==train_data[:, -1])
        if len(idx) == 0:
            continue

        fingers = train_data[idx[:,0],:]
        # acquire rssi of fingers



        rssis = fingers[:,1:-3]
        # defien avg_rssi and avg_num
        avg_rssi = np.zeros((rssis.shape[1]))
        avg_num = np.zeros((rssis.shape[1]))
        for j,value in enumerate(rssis):
            value_idx = np.argwhere(value != 0)
            avg_rssi[value_idx] = avg_rssi[value_idx] + value[value_idx]
            avg_num[value_idx] =  avg_num[value_idx] + 1
        value_idx = np.argwhere(avg_rssi != 0 )
        avg_rssi[value_idx] = avg_rssi[value_idx] / avg_num[value_idx]
        refrence_rssi[i, 0:-3] = avg_rssi


    return [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Sequential_test_data, Graident_test_data,Sequential_train_datas,Graident_train_datas,refrence_rssi]




def Windows_Split_Module(train_data, TIME_STEP):


    for i in range(train_data.shape[0]):

        start = i

        end = i + TIME_STEP

        if end > train_data.shape[0]:
            batch = np.concatenate((train_data[start:, :], train_data[0:end-train_data.shape[0], :]), axis=0)
        else:
            batch = train_data[start:end, :]

        batch = batch.reshape(1, TIME_STEP, -1)
        if not "batchs" in locals().keys():

            batchs = batch

        else:
            batchs = np.concatenate((batchs, batch), axis=0)

    return batchs


def Graident_Generator(train_data):
    Graident_train_data = train_data.copy()
    for i, item in enumerate(Graident_train_data):
        # print(item)
        end_Node = item[-1,:]
        end_Node_prints = end_Node[1:-3]
        for j in range(TIME_STEP-1):
            Graident_train_data[i, j, 1:-3] = Graident_train_data[i,j,1:-3] - end_Node_prints

    return Graident_train_data


if __name__ == '__main__':

    # Data Type
    REF1 = "ref_1.txt"
    REF2 = "ref_2.txt"


    WIFI2 = "wifi2.csv"
    WIFI3 = "wifi3.csv"


    datatypes = [WIFI3, WIFI2]

    # Select Person's Trajectory: 0 is David, 1 is You
    path_idx = 1

    datatype = WIFI3

    # Select test Tajectory
    test_idx = 1

    # Select grid_size
    seperation_id = 8


    # define Hyper Parameters
    TIME_STEP = 4  # rnn time step
    INPUT_SIZE = 468  # rnn input size
    LR = 0.0005  # learning rate
    BATCH_SIZE = 100

    persons = [0, 1]

    # read data and build loader
    # Input: path_idx : Id of  Person
    #        datatype : Data Type
    #        test_idx : Id of Test Data from path_idx
    #       seperation_id: Id of Grid Size
    #       TIME_STEP: Time step for LSTM
    #       INPUT_SIZE: the number of APs
    #
    # Ouput: train_x: features of all trian data [RSSI of all APS]
    #        train_y_c:Id of Position  accord to variable all_refences for trian data
    #       train_y_r:Position of trian data[x,y] for trian data
    #       test_x: features of all test data [RSSI of all APS]
    #       test_y_c: Id of Position accord to variable all_refences for test data
    #       test_y_r:Position of trian data[x,y] for test data
    #       all_refences:Position and Id table
    #       Sequential_test_data: Sequential of test data consisted of [Time,RSSI...,Position,Id of Position], shape=[BATCH_SIZE,TIEM_STEP,INPUT_SIZE]
    #       Graident_test_data: Graident of test datas consisted of [Time,DRSSI...,Position,Id of Position], shape=[BATCH_SIZE,TIEM_STEP,INPUT_SIZE]
    #       Sequential_train_datas: Sequential of train data consisted of [Time,RSSI...,Position,Id of Position], shape=[BATCH_SIZE,TIEM_STEP,INPUT_SIZE]
    #       Graident_train_datas: Graident of train datas consisted of [Time,DRSSI...,Position,Id of Position], shape=[BATCH_SIZE,TIEM_STEP,INPUT_SIZE]
    #       refrence_rssi:RSSIS for every RPs

    # [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Sequential_test_data, Graident_test_data, Sequential_train_datas, Graident_train_datas,refrence_rssi]= read_data(path_idx, datatype, test_idx, seperation_id, TIME_STEP,INPUT_SIZE)
    # you_wifi2
    [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Graident_test_data,Sequential_test_data, Graident_train_datas,Sequential_train_datas,  refrence_rssi] = read_data(persons, datatypes, test_idx, seperation_id, TIME_STEP, INPUT_SIZE)
    # # you_wifi3
    # [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Graident_test_data,Sequential_test_data, Graident_train_datas,Sequential_train_datas,  refrence_rssi] = read_data(path_idx, datatype, test_idx, seperation_id, TIME_STEP, INPUT_SIZE)
    # # david_wifi2
    # [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Graident_test_data,Sequential_test_data, Graident_train_datas,Sequential_train_datas,  refrence_rssi] = read_data(path_idx, datatype, test_idx, seperation_id, TIME_STEP, INPUT_SIZE)
    # # david_wifi3
    # [train_x, train_y_c, train_y_r, test_x, test_y_c, test_y_r, all_refences, Graident_test_data,Sequential_test_data, Graident_train_datas,Sequential_train_datas,  refrence_rssi] = read_data(path_idx, datatype, test_idx, seperation_id, TIME_STEP, INPUT_SIZE)
    #

    #concatanate

    # Data.TensorDataset(x, y)
    # loader = Data.DataLoader(
    #
    # dataset= torch.cat((Sequential_train_datas.unsqueeze(1),Graident_train_datas.unsqueeze(1)),dim=1),
    # batch_size = BATCH_SIZE,
    # shuffle=True
    #
    # )



    CLASS = all_refences.shape[0]


    train_x = torch.from_numpy(train_x).to(torch.float32).cuda()
    train_y_c = torch.from_numpy(train_y_c).to(torch.int64).cuda()
    train_y_r = torch.from_numpy(train_y_r).to(torch.float32).cuda()
    test_x = torch.from_numpy(test_x).to(torch.float32).cuda()
    test_y_c = torch.from_numpy(test_y_c).to(torch.int64).cuda()
    test_y_r = torch.from_numpy(test_y_r).to(torch.float32).cuda()
    all_refences = torch.from_numpy(all_refences).cuda()

    Sequential_test_data = torch.from_numpy(Sequential_test_data).cuda()
    Graident_test_data = torch.from_numpy(Graident_test_data).cuda()

    Sequential_train_datas = torch.from_numpy(Sequential_train_datas).cuda()
    Graident_train_datas = torch.from_numpy(Graident_train_datas).cuda()
    x=torch.cat((Sequential_train_datas[:, :, 1:-3].unsqueeze(1), Graident_train_datas[:, :, 1:-3].unsqueeze(1)), dim=1).cuda()
    y=torch.cat((Sequential_train_datas[:, :, -3:].unsqueeze(1), Graident_train_datas[:, :, -3:].unsqueeze(1)), dim=1).cuda()
    torch_dataset =Data.TensorDataset(x,y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    refrence_rssi = torch.from_numpy(refrence_rssi).to(torch.float32)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # define parameters
        self.hideen_size = 60
        self.num_layters = 1

        self.RSSI2hidden = nn.Sequential(
            nn.BatchNorm1d(INPUT_SIZE, momentum=0.5),
            nn.Linear(INPUT_SIZE, self.hideen_size*4),
            nn.BatchNorm1d(self.hideen_size*4, momentum=0.5),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hideen_size*4, self.hideen_size*2)
        )


        self.DRSSI2hidden = nn.Sequential(
            nn.BatchNorm1d(INPUT_SIZE, momentum=0.5),
            nn.Linear(INPUT_SIZE, self.hideen_size * 4),
            nn.BatchNorm1d(self.hideen_size * 4, momentum=0.5),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(self.hideen_size*4, self.hideen_size*2)
        )


        self.encoder = nn.LSTM(

            input_size=self.hideen_size*2,
            hidden_size=self.hideen_size,
            num_layers=self.num_layters,
            batch_first=True,
            # dropout=0.2,
        )

        # # Attention Model 1
        # self.weights_layer = nn.Sequential(
        #
        #     nn.Linear(self.hideen_size*(TIME_STEP+1), 90),
        #     nn.BatchNorm1d(90, momentum=0.5),
        #     nn.BatchNorm1d(90, momentum=0.5),
        #     torch.nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(90,5)
        #
        # )

        # Attention Model 3
        self.weights_layer = nn.Sequential(

            nn.Linear(self.hideen_size*2, 90),
            nn.BatchNorm1d(90, momentum=0.5),
            torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(90, 1)

        )


        self.decoder = nn.LSTM(

            input_size=self.hideen_size*5,
            hidden_size=self.hideen_size,
            num_layers=self.num_layters,
            batch_first=True,
            # dropout=0.2,

        )
        self.predicter = nn.Sequential(

            nn.Linear(self.hideen_size * 6, 240),
            nn.BatchNorm1d(240, momentum=0.5),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(240, CLASS)

        )
    def Predicter(self,y_hat_i_1, S_h, TEMP_RSSI,C_i):


        out = self.predicter( torch.cat((y_hat_i_1, S_h.squeeze(0), TEMP_RSSI, C_i.squeeze(1)), dim=1))
        return out
    def Encoder(self,DRSSI):

        # encode module
        TEMP_DRSSI = torch.tensor([]).cuda()

        for i in range(TIME_STEP-1):

            TEMP_DRSSI = torch.cat((TEMP_DRSSI,self.DRSSI2hidden(DRSSI[:, i, :]).unsqueeze(1)), dim=1)

        TEMP_DRSSI = torch.cat((TEMP_DRSSI, self.RSSI2hidden(DRSSI[:, -1, :]).unsqueeze(1)), dim=1)

        DRSSI = TEMP_DRSSI
        H, (h_, c_) = self.encoder(DRSSI, None)

        S_h = h_
        S_c = c_

        return H,S_h,S_c

    def Attention(self,H,S_h):

        weights = torch.tensor([]).cuda()
        # attention method 1
        # weights = self.weights_layer(torch.cat(( S_h.view(-1,1,self.hideen_size), H), 1).view(-1,(TIME_STEP+1)*self.hideen_size))

        # attention method 3
        for i in range(TIME_STEP):
            TEMP_WEIGHT = self.weights_layer(torch.cat(( S_h[0,:,:], H[:,i,:]), 1))
            weights = torch.cat((weights, TEMP_WEIGHT), dim=1)

        # attention method 2
        #TEMP_WEIGHTS = torch.tensor([]).cuda()
        # for i in range(TIME_STEP):
        #     TEMP_WEIGHT = torch.sum(torch.mul(H[:, i, :], S_h[0, :, :]), dim=1) / (torch.sqrt(torch.sum(H[:, i, :] * H[:, i, :], dim=1)) * torch.sqrt(torch.sum(S_h[0, :, :] * S_h[0, :, :], dim=1)))
        #
        #     TEMP_WEIGHTS = torch.cat((TEMP_WEIGHTS, TEMP_WEIGHT.view(-1, 1)), dim=1)
        #
        # weights = TEMP_WEIGHTS

        weights = F.softmax(weights)

        C_i = torch.bmm(weights.view(-1, 1, TIME_STEP), H)

        return C_i

    def Decoder(self, y_hat_i_1,TEMP_RSSI,C_i,S_h,S_c):

        # Decoder
        # Input:y_hat_i_1.shape=[batch_size,input_size]
        #       C_i.shape=batch_size,time_step,input_size => batch_size,input_size      ===============>[batch_size,time_step,input_size]
        #       TEMP_RSSI.shape = [batch_size,input_size ]

        # Input for hidden:S_h.shape = [Time_step,batch_size,input_size]
        #                 S_c.shape = [Time_step,batch_size,input_size]

        # Output:S.shape=[batch_size,time_step,input_size]
        #       S_h.shape = [Time_step,batch_size,input_siz]
        #       S_c.shape = [Time_step,batch_size,input_siz]

        S, (S_h, S_c) = self.decoder(torch.cat((y_hat_i_1, TEMP_RSSI, C_i.view(-1, self.hideen_size)), dim=1).unsqueeze(1), (S_h, S_c))

        return S,S_h,S_c


    def Prediction(self,H,S_h,S_c,y_hat_i_1,RSSI,result,refrence_rssi):

        for j in range(TIME_STEP):

            C_i = self.Attention(H,S_h)  # H.shape(batch_size,time_step,input_size) S_h.shape=[time_step,batch_sizr,input_size]
            # input:y_hat_i_1.shape = [batch_size,input_size]
            # output: batch_size,input_size
            y_hat_i_1 = self.RSSI2hidden(y_hat_i_1)

            # input:RSSI.shape=[batch_size,time_step,input_size] =>[batch_size,j,input_size]
            # output: batch_size,input_size
            TEMP_RSSI = self.RSSI2hidden(RSSI[:, j, :])

            # input: y_hat_i_1.shape=  [batch_size,j,input_size]
            #       TEMP_RSSI.shape =  [batch_size,j,input_size]
            #       C_i.shape = [batch_size,time_step,input_size]
            #       S_h.shape = [time_step,batch_size,input_size]
            #       S_c.shape = [time_step,batch_size,input_size]
            # output:
            #      S_h.shape = [time_step,batch_size,input_size]
            #      S_c.shape = [time_step,batch_size,input_size]
            #      S =[batch_size,time_step,input_size]

            S, S_h, S_c = self.Decoder(y_hat_i_1, TEMP_RSSI, C_i, S_h, S_c)

            # result
            out = self.Predicter(y_hat_i_1, S_h, TEMP_RSSI, C_i)

            result = torch.cat((result, out.unsqueeze(1)), dim=1)
            out_c = F.log_softmax(out)


            topv, topi = out_c.topk(1)

            y_hat_i_1 = refrence_rssi[topi[:, 0], 0:-3].cuda()

        return result

    def recursion(self, H, S_h, S_c, y_hat_i_1, RSSI, result, refrence_rssi, j, idx, LIST):

        if j <= (TIME_STEP-1):

            C_i = self.Attention(H,S_h)  # H.shape(batch_size,time_step,input_size) S_h.shape=[time_step,batch_sizr,input_size]
            # input:y_hat_i_1.shape = [batch_size,input_size]
            # output: batch_size,input_size
            y_hat_i_1 = self.RSSI2hidden(y_hat_i_1)

            # input:RSSI.shape=[batch_size,time_step,input_size] =>[batch_size,j,input_size]
            # output: batch_size,input_size
            TEMP_RSSI = self.RSSI2hidden(RSSI[:, j, :])

            # input: y_hat_i_1.shape=  [batch_size,j,input_size]
            #       TEMP_RSSI.shape =  [batch_size,j,input_size]
            #       C_i.shape = [batch_size,time_step,input_size]
            #       S_h.shape = [time_step,batch_size,input_size]
            #       S_c.shape = [time_step,batch_size,input_size]
            # output:
            #      S_h.shape = [time_step,batch_size,input_size]
            #      S_c.shape = [time_step,batch_size,input_size]
            #      S =[batch_size,time_step,input_size]

            S, S_h, S_c = self.Decoder(y_hat_i_1, TEMP_RSSI, C_i, S_h, S_c)

            # result
            out = self.Predicter(y_hat_i_1, S_h, TEMP_RSSI, C_i)

            out_c = F.softmax(out)

            # True for test ,False for train
            topv, topi = out_c.topk(k)
            y_hat_i_1_s = []


            for i in range(1, k + 1):

                # save a and  b to tree
                LIST[idx*k+i, 0] = topi[:, i-1]

                LIST[idx*k+i, 1] = topv[:, i-1]

                # acqiure y_hat_i_1
                y_hat_i_1_s.append(refrence_rssi[topi[:, i-1], 0:-3].cuda())

                self.recursion(H, S_h, S_c, refrence_rssi[topi[:, i-1], 0:-3].cuda(), RSSI, result, refrence_rssi, j + 1, idx*k+i, LIST)


            return 0
        else:
            return 0



    def forward(self, RSSI,DRSSI,refrence_rssi,k,sign):

        #Input: DRSSI.shape=[batch_size tiem_step input_size]
        #Output: H.shape = [batch_size,time_step,input_size]
        #       S_h.shape = [time_step,batch_sizr,input_size]
        #       S_c.shape = [time_step,batch_sizr,input_size]
        H, S_h, S_c = self.Encoder(DRSSI)


        BATCH_SIZE = RSSI.shape[0]

        # init y_hat_i_1 with 0
        y_hat_i_1 = torch.zeros(BATCH_SIZE,INPUT_SIZE).cuda()

        # attention module
        result = torch.tensor([]).cuda()


        if sign:
            # prediction

            return self.Prediction(H, S_h, S_c, y_hat_i_1, RSSI, result, refrence_rssi)

        else:
            #train

            j = 0
            idx = 0

            NUM = int(math.pow(k, 0)*(1-math.pow(k, TIME_STEP+1))/(1-k))
            LIST = torch.ones((NUM, 2, BATCH_SIZE)).cuda() * -1
            self.recursion(H, S_h, S_c, y_hat_i_1, RSSI, result, refrence_rssi, j,idx,LIST)

        return LIST





rnn = torch.load("time_2017__03.pkl")
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = torch.nn.CrossEntropyLoss()

h_state = None  # for initial hidden state

init_value = 100000

train_size = train_x.shape[0]
min = 1000
LOSS = 0
num = 0
avg_loss = []
k = 3



for epoch in range(2000):
    for step, (batch_x, batch_y) in enumerate(loader):
        start_time = time.time()


        err = 0
        pre = torch.tensor([]).cuda()
        all_t_y_r = torch.tensor([]).cuda()

        test_x_RSSI = Sequential_test_data[:,:,1:-3]
        test_x_DRSSI = Graident_test_data[:,:,1:-3]
        testy = Sequential_test_data[:,:,-1]
        t_y_r =  Sequential_test_data[:,:,-3:-1]
        # t_x_RSSI = test_x_RSSI.view(:, TIME_STEP, INPUT_SIZE)
        # t_x_DRSSI = test_x_DRSSI.view(BATCH_SIZE, TIME_STEP, INPUT_SIZE)
        # t_y = testy.view(BATCH_SIZE, TIME_STEP, 1)

        #  tree opt True mean train but False mean test
        sign = False

        out = rnn(test_x_RSSI.float(),test_x_DRSSI.float(),refrence_rssi.float(),k,sign)
        # pre =torch.cat((pre,out),dim=0)
        # all_t_y_r = torch.cat((all_t_y_r,t_y_r),dim=0)


        if sign==False:
            # the number of solutions
            NUM = int(math.pow(k,TIME_STEP))

            # list results and get highest score from NUM results

            # save err for every signal solution
            accumulate_err = torch.zeros(NUM,t_y_r.shape[0]).cuda()

            # save probability for every signal solution
            accumulate_prob = torch.zeros(NUM,t_y_r.shape[0]).cuda()

            # save Nodes for every signal solution
            accumulate_node = torch.zeros(NUM,t_y_r.shape[0], TIME_STEP).cuda()

            # First solution on leaf node = a1(1-q^n)/(1-q)
            start= int(math.pow(k,0)*(1-math.pow(k,TIME_STEP))/(1-k))

            for i in range(NUM):
                # acquire idx of start node
                idx = i + start
                # list all node of a tree path
                for j in range(TIME_STEP):

                    accumulate_prob[i,:] = accumulate_prob[i,:] + out[idx, 1,:]

                    accumulate_node[i,:,TIME_STEP-1-j] = out[idx,0,:]
                    # acquire idx of father node
                    idx = int((idx - 1)/k)
            # save sorted score nodes
            higest_score_prediction = torch.zeros(t_y_r.shape[0],NUM,TIME_STEP)
            # save
            higest_accrate_predition = torch.zeros(t_y_r.shape[0],TIME_STEP)
            avg_errs = 0
            avg_errs_hights = 0
            all_errs = []
            for i in range(t_y_r.shape[0]):

                # index for prob
                tree_path_idx_pro = torch.argsort(accumulate_prob[:, i],descending=True)

                # all sequential solutions for node i
                higest_score_prediction[i,:,:] = accumulate_node[tree_path_idx_pro, i, :]
                # avg for the number of solutions
                n =4

                # acquire n y_hat
                n_y_hat = all_refences[higest_score_prediction[i, 0:n, :].to(torch.int64), 0:2]
                # avg
                y_hat = torch.sum(n_y_hat,dim =0 )/n
                if i ==0:
                    Y_hat = y_hat.view(1, -1, 2)
                else:
                    Y_hat = torch.cat((Y_hat, y_hat.view(1, -1, 2)), dim=0)
                n_errs_hights = y_hat - t_y_r[i]
                n_errs_hights = torch.pow(n_errs_hights, 2)
                n_errs_hights = torch.sum(n_errs_hights, dim = 1)
                n_errs_hights = torch.sqrt(n_errs_hights)
                n_errs_hights = torch.sum(n_errs_hights)/TIME_STEP

                avg_errs_hights = avg_errs_hights + n_errs_hights
                all_errs.append(n_errs_hights.item())
                # lowest err
                errs = torch.sum(torch.sqrt(torch.sum(torch.pow(all_refences[accumulate_node[:, i, :].to(torch.int64), 0:2] - t_y_r[i], 2), dim=2)), dim=1) / TIME_STEP
                tree_path_idx_err = torch.argmin(errs)
                err = errs[tree_path_idx_err]
                higest_accrate_predition[i,:] = accumulate_node[tree_path_idx_err, i, :]
                avg_errs =avg_errs + err

            avg_errs = avg_errs / t_y_r.shape[0]
            avg_errs_hights = avg_errs_hights / t_y_r.shape[0]

            # print(avg_errs_hights)
            end_time = time.time()
            if min > avg_errs_hights:
                min_Y_hat = Y_hat
                min_all_errs = all_errs
                # min_errs =
                min = avg_errs_hights

            # print(avg_errs)
            # print(higest_accrate_predition)
            # print(higest_score_prediction)
            print("avg_errs",avg_errs,"avg_errs_hights",avg_errs_hights,"min",min,'cost time',end_time-start_time)

        else:

            # err = err + sum(torch.sqrt(torch.sum(torch.pow(t_y_r -out,2),dim = 1)))
            topi,topc=out.topk(1)


            pre = torch.cat((pre,topc.float()))

            all_t_y_r = torch.cat((all_t_y_r.float(),t_y_r.float()))

            pre = pre.to(torch.int64)
            errs = torch.sqrt(torch.sum(torch.pow(all_refences[topc.flatten(), 0:2] - t_y_r.view(-1, 2), 2), dim=1))
            err = torch.sum(errs) / (TIME_STEP*t_y_r.shape[0])

            # err = torch.sum(torch.sqrt(torch.sum(torch.pow(pre - all_t_y_r,2),1))) / (Sequential_test_data.shape[0]*Sequential_test_data.shape[1])
            # plt.figure()
            # plt.scatter(all_refences[pre.view(670),0].detach().numpy(), all_refences[pre.view(670),0:1].detach().numpy())
            # plt.scatter(all_t_y_r[:, 0].detach().numpy(), all_t_y_r[:, 1].detach().numpy())

            if min > err:
                min = err
                min_pre = pre
                min_point = all_refences[topc.flatten(), 0:2]
                # torch.save(rnn, 'Navigation_09_softmax.pkl')
            end_time = time.time()
            if step % 20 == 0:
                print("Epoch", epoch,  "err", err, "minerr", min, 'cost time', end_time-start_time)
