#TechVidvan
# load all required libraries for Dog's Breed Identification Project
import cv2
import numpy as np 
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

#read the csv file
df_labels = pd.read_csv("path")
#store training and testing images folder location
train_file = 'path/' #path đến tập train
test_file = 'path/' #path đến tập test 

print("Total number of unique Dog Breeds: ", len(df_labels.breed.unique()))

num_breeds = 60
im_size = 224
batch_size = 64
encoder = LabelEncoder()

#get only 60 unique breeds record 
breed_dict = list(df_labels['breed'].value_counts().keys()) 
new_list = sorted(breed_dict,reverse=True)[:num_breeds*2+1:2]
#change the dataset to have only those 60 unique breed records
df_labels = df_labels.query('breed in @new_list')

#create new column which will contain image name with the image extension
df_labels['img_file'] = df_labels['id'].apply(lambda x: x + ".jpg")


train_x = np.zeros((len(df_labels), im_size, im_size, 3), dtype='float32')

#enumerate() là hàm thêm 1 bộ đếm vào 1 đối tượng có thể lặp lại 
for i, img_id in enumerate(df_labels['img_file']):
    # đọc tệp hình ảnh và chuyển đổi sang định dạng số
    # điều chỉnh kích thước tất cả hình ảnh thành một chiều, tức là 224x224
    # chúng ta sẽ nhận được mảng có hình dạng là
    # (224,224,3) trong đó 3 là các lớp kênh RGB
    img = cv2.resize(cv2.imread(train_file+img_id,cv2.IMREAD_COLOR),((im_size,im_size)))
    img_array = preprocess_input(np.expand_dims(np.array(img[...,::-1].astype(np.float32)).copy(), axis=0))
    #cập nhật biến train_x bằng phần tử mới
    train_x[i] = img_array

    # Đây sẽ là mục tiêu cho mô hình.
#convert tên giống sang định dạng số
train_y = encoder.fit_transform(df_labels["breed"].values)

#split tập dữ liệu theo tỷ lệ 80:20.
# 80% cho mục đích đào tạo và 20% cho mục đích thử nghiệm
x_train, x_test, y_train, y_test = train_test_split(train_x,train_y,test_size=0.2,random_state=42)

#Tăng hình ảnh bằng cách sử dụng lớp ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.25,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
 
#generate hình ảnh cho bộ đào tạo
train_generator = train_datagen.flow(x_train, 
                                     y_train, 
                                     batch_size=batch_size)
 
#same process cho bộ Thử nghiệm cũng bằng cách khai báo phiên bản
test_datagen = ImageDataGenerator()
 
test_generator = test_datagen.flow(x_test, 
                                     y_test, 
                                     batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(im_size, im_size, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(120,activation='softmax'))

model.summary()

#epochs for model training and learning rate for optimizer
epochs = 20
learning_rate = 1e-3
 
#using RMSprop optimizer to compile or build the model
#optimizer = RMSprop(learning_rate=learning_rate,rho=0.9)
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
 
#fit the training generator data and train the model
hist = model.fit(train_generator,
                 steps_per_epoch= x_train.shape[0] // batch_size,
                 epochs= epochs,
                 validation_data= test_generator,
                 validation_steps= x_test.shape[0] // batch_size)
 
#Save the model for prediction
model.save("model")
