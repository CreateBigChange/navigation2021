import  numpy as np
import  pandas as pd

path="D:\developmentTool\Code\DATA\SCI 10 DATA _v1_150906\Trajectory_You\\20150903010338\ios_ble_1.csv"
data=pd.read_csv(path)
mac=np.array(list(set(data.iloc[:,1])))

channel=np.ones((mac.shape[0],1))*(-1)

for i  in range(data.shape[0]):
    idx= np.argwhere(data.iloc[i,1]==mac)[0,0]

    if channel[idx,0]==-1:
        channel[idx, 0]=data.iloc[i,1]
    elif channel[idx, 0]==data.iloc[i,1]:
        print('normal')
    else:
        print(i)
        print(data.iloc[i,1])
        print("----")
        print(channel[i,0])
        print('------')
        print(data.iloc[i,3])