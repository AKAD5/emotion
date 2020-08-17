# 导入模块
import keras
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

# 定义参数
batch_siz=128
num_class=2
nb_epoch=50
img_size=48
root_path="E:/python3/jupyter notebooj/image_recognition/keras/dj"

# CNN
class model:
    def __init__(self):
        self.model=None

    def cnn_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32,(3,3)),strides=1,padding="same")
        self.model.add(Activation("relu"))
        self.model.add(Conv2D(32,(5,5)),padding="same")
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(32,(3,3)),padding="same")
        seld.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(64,(5,5)),padding=(2,2))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(2048))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(1024))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        self.model.add(num_class)
        self.model.add(Activation("softmax"))
        self.model.summary()

    def train_model(self):
        sgd=SGD(lr=0.001,decay=0.000001,momentum=0.9,nesterov=True)
        self.model.complie(loss="categorical_crossentropy",optimize=sgd,metrics=["accuracy"])

        train_
