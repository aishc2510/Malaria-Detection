from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

Pcell=os.listdir('C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images\\Parasitized')
ucell=os.listdir('C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images\\Uninfected')
print("Parasitized cell:",len(Pcell))
print("Uninfcted cell:",len(ucell))

width = 68
height = 68

data = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)

trainData = data.flow_from_directory(directory='C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images',
                                           target_size=(width,height),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='training')

trainData.class_indices

valData = data.flow_from_directory(directory='C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images',
                                           target_size=(width,height),
                                           class_mode = 'binary',
                                           batch_size = 16,
                                           subset='validation')

model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(width,height,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit_generator(generator=trainData,
                            steps_per_epoch=len(trainData),
                            epochs=5,
                            validation_data=valData ,
                            validation_steps=len(valData )
                           )

tf.keras.models.save_model(model,'C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\model.hdf5')









