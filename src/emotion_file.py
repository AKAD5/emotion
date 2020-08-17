import pandas as pd
import numpy as np
from PIL import Image
import os

path=os.path.abspath(".")
emotion_Yes=path+"\\image\\1"
emotion_No=path+"\\image\\0"

Y=[]
for i in os.listdir(emotion_Yes):
    im=Image.open(emotion_Yes+"\\"+i)
    im=im.resize((48,48))
    im=im.convert("L")
    Y.append(np.array(im))

N=[]
for i in os.listdir(emotion_No):
    im=Image.open(emotion_No+"\\"+i)
    im=im.resize((48,48))
    im=im.convert("L")
    N.append(np.array(im))

d=[]
for i in Y:
    d1=[]
    for j in i:
        for n in j:
            d1.append(n)
    d.append(str(d1).replace("[","").replace("]","").replace(",",""))

e=[]
for i in N:
    e1=[]
    for j in i:
        for n in j:
            e1.append(n)
    e.append(str(e1).replace("[","").replace("]","").replace(",",""))

dat1=pd.DataFrame({"emotion":[1]*len(d),"pixels":d})
jh1=["Training"]*int(dat1.shape[0]*0.7)+["PrivateTest"]*(int(dat1.shape[0]*0.2))+["PublicTest"]*(int(dat1.shape[0]-int(dat1.shape[0]*0.7)-int(dat1.shape[0]*0.2)))
#训练集、测试集、验证集=70%，20%，10%
dat1["Usage"]=jh1

dat2=pd.DataFrame({"emotion":[0]*len(e),"pixels":e})
jh2=["Training"]*int(dat2.shape[0]*0.7)+["PrivateTest"]*(int(dat2.shape[0]*0.2))+["PublicTest"]*(int(dat2.shape[0]-int(dat2.shape[0]*0.7)-int(dat2.shape[0]*0.2)))
dat2["Usage"]=jh2

data_x=pd.concat([dat1,dat2],ignore_index=True)
data_x.to_csv(path+"\\emotion_file.csv",index=False)

print(data_x.shape)
