#图像安装框选大小进行裁切保存
import pandas as pd
import os
import json
from PIL import Image

path=os.path.abspath(".")
path_image=path+"\\dat_20191202"
for i,j in zip(os.listdir(path_image),range(len(os.listdir(path_image)))):
    if i.endswith("json"):
        with open(path_image+"\\"+i,"r") as f:
            data2=json.load(f)
            path_i=data2["asset"]["name"]
            labe=data2["regions"][0]["tags"][0]
            box=(data2["regions"][0]["points"][0]["x"],data2["regions"][0]["points"][0]["y"],data2["regions"][0]["points"][2]["x"],data2["regions"][0]["points"][2]["y"])
            im=Image.open(path_image+"\\"+path_i).crop(box)
            try:
                im.save(path+"\\image\\0\\"+str(j)+".jpg")
            except:
                print("格式错误")
