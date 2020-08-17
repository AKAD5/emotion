import numpy as np
import cv2
import sys
import json
import time
import os
from keras.models import model_from_json
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
root_path=os.path.abspath(".")
model_path = root_path + '/model_50/'  # '/model_0.7/'
img_size = 48

# emotion_labels = ["disgust",'happy']
emotion_labels = ["happy","disgust"] #model_50
num_class = len(emotion_labels)
json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(model_path + 'model_weight.h5')

# def predict_emotion(face_img):
#     face_img = face_img * (1. / 255)
#     resized_img = cv2.resize(face_img, (img_size, img_size))  # ,interpolation=cv2.INTER_LINEAR
#     rsz_img = []
#     rsh_img = []
#     results = []
#     rsz_img.append(resized_img[:, :])  # resized_img[1:46,1:46]
#     # rsz_img.append(resized_img[2:45, :])
#     # rsz_img.append(cv2.flip(rsz_img[0], 1))
#     # rsz_img.append(cv2.flip(rsz_img[1],1))
#     i = 0
#     for rsz_image in rsz_img:
#         rsz_img[i] = cv2.resize(rsz_image, (img_size, img_size))
#         i += 1
#     for rsz_image in rsz_img:
#         rsh_img.append(rsz_image.reshape(1, img_size, img_size, 1))
#     i = 0
#     for rsh_image in rsh_img:
#         list_of_list = model.predict_proba(rsh_image, batch_size=32, verbose=1)  # predict
#         result = [prob for lst in list_of_list for prob in lst]
#         results.append(result)
#     return results

def predict_emotion(face_img):
    img_size=48
    face_img = face_img * (1. / 255)
    resized_img = cv2.resize(face_img, (img_size, img_size))  # ,interpolation=cv2.INTER_LINEAR
    results = []
    rsh_img=resized_img.reshape(1,img_size, img_size, 1)
    list_of_list = model.predict_proba(rsh_img, batch_size=32, verbose=1)  # predict
    result = [prob for lst in list_of_list for prob in lst]
    results.append(result)

    return results

def face_detect(image_path):
    cascPath = 'C:/ProgramData/Anaconda3/lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml'
    faceCasccade = cv2.CascadeClassifier(cascPath)
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCasccade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(10,10)
    )

    return faces, img_gray, img


if __name__ == '__main__':
    images = []
    flag = 0
    A=input("输入文件夹名称：")
    # if len(sys.argv) != 0:
    dir = root_path +"\\"+ A
    if os.path.isdir(dir):
        files = os.listdir(dir)
        for file in files:
            if file.endswith('jpg') or file.endswith('png') or file.endswith('PNG') or file.endswith('JPG'):
                images.append(dir + '\\' + file)
    else:
        images.append(dir)
    for image in images:
        faces1, img_gray, img = face_detect(image)
        if faces1==():
            faces1=np.array([[0,0,img_gray.shape[1],img_gray.shape[0]]])
        faces=[]
        for i in faces1:
            x,y,w,h=i
            if w >= max(faces1[:,2])*0.8:
                j=(x,y,w,h)
                faces.append(j)
        spb = img.shape
        sp = img_gray.shape
        height = sp[0]
        width = sp[1]
        size = 600
        # if flag == 0:
        emo = ""
        face_exists = 0
        for (x, y, w, h) in faces:
            face_exists = 1
            face_img_gray = img_gray[y:y + h, x:x + w]
            results = predict_emotion(face_img_gray)  # face_img_gray
            result_sum = np.array([0]*num_class)
            for result in results:
                print(result)
                result_sum = result_sum + np.array(result)
            happy,disgust= result_sum
                # 输出所有情绪的概率
            print(result_sum)
            print('happy:', happy,' disgust:', disgust)
            label = np.argmax(result_sum)
            emo = emotion_labels[label]
            print('Emotion : ', emo)
                # 输出最大概率的情绪
            t_size = 2
            ww = int(spb[0] * t_size / 300)
            www = int((w + 10) * t_size / 100)
            www_s = int((w + 20) * t_size / 100) * 2 / 5
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), ww)
                # cv2.putText(img, emo, (x + 2, y + h - 2), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                #             www_s, (255, 0, 255), thickness=www, lineType=1)
            cv2.putText(img, emo, (x+2, y+h-10),cv2.FONT_HERSHEY_TRIPLEX,
                            www_s, (255, 0, 0), thickness=www, lineType=1)
                # img_gray full face     face_img_gray part of face   #cv2.FONT_HERSHEY_SIMPLEX
        if face_exists:
            cv2.HoughLinesP
            cv2.namedWindow(emo, 0)
            cent = int((height * 1.0 / width) * size)
            cv2.resizeWindow(emo, (size, cent))
            cv2.imshow(emo, img)
            k = cv2.waitKey(0)#窗口
            cv2.destroyAllWindows()
            if k & 0xFF == ord('q'):
                break
        # elif flag == 1:
        #     img = cv2.imread(image)
        #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     results = predict_emotion(img_gray)  # face_img_gray
        #     result_sum = np.array([0]*num_class)
        #     for result in results:
        #         result_sum = result_sum + np.array(result)
        #         print(result)
        #     disgust,happy= result_sum
        #     # 输出所有情绪的概率
        #     print(result_sum)
        #     print('disgust:', disgust, ' happy:', happy)
        #     label = np.argmax(result_sum)
        #     emo = emotion_labels[label]
        #     print('Emotion : ', emo)
        #     输出最大概率的情绪
        #
        #     # img_gray full face     face_img_gray part of face
        #     cv2.HoughLinesP
        #     # cv2.imwrite('./'+emo+'.jpg',face_img_gray)
        #     cv2.namedWindow(emo, 0)
        #     size = 400
        #     cent = int((height * 1.0 / width) * size)
        #     cv2.resizeWindow(emo, (size, cent))
        #
        #     cv2.imshow(emo, img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
