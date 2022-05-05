from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


testing_path="C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images\\Parasitized\\C39P4thinF_original_IMG_20150622_105102_cell_90.png"
img=image.load_img(testing_path,target_size=(68,68))
plt.imshow(img)
model=tf.keras.models.load_model("C:\\Users\\Abhishek\\Desktop\\Malaria-Infected-Cell-Classification-main\\models\\mymodel.hdf5")

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
val=model.predict(images)
if val==0:
    plt.title("Paracitized")
else:
    plt.title("Uninfected")
    
    testing_path="C:\\Users\\Abhishek\\Desktop\\MALARIA_DETECTION\\cell_images\\cell_images\\Parasitized\\Uninfected\\C45P6ThinF_IMG_20151130_160135_cell_115.png"
img=image.load_img(testing_path,target_size=(68,68))
plt.imshow(img)

x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])
val=model.predict(images)
if val==0:
    plt.title("Paracitized")
else:
    plt.title("Uninfected")