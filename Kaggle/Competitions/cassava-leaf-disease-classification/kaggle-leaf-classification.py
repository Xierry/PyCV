import pandas as pd

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import os
import cv2
# import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, Xception
from tensorflow.keras.optimizers import Adam


input_dir = "../input/cassava-leaf-disease-classification"
train_images_path = os.path.join(input_dir,"train_images")
test_images_path = os.path.join(input_dir,'test_images')

train_image_fmt = os.path.join(input_dir,"train_images/{}")
test_image_fmt = os.path.join(input_dir,'test_images/{}')

train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')

BATCH_SIZE = 8 # Mini-Batch Gradient Descent
STEPS_PER_EPOCH = len(train)*0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train)*0.2 / BATCH_SIZE
EPOCHS = 20
TARGET_SIZE = 350

with open(os.path.join(input_dir,"label_num_to_disease_map.json")) as f:
    # class_mapping = json.load(f)
	class_mapping2 ={int(k):v for k,v in json.load(f).items()}


train.label = train.label.astype('str')

train_datagen = ImageDataGenerator(
	validation_split = 0.2,
	rotation_range = 45, # 旋转范围
	zoom_range = 0.3, # 缩放范围
	horizontal_flip = True,
    vertical_flip = True, fill_mode = 'nearest', 
    shear_range = 0.1, # 透视变换的范围
    width_shift_range = 0.1, # 水平平移范围
 	height_shift_range = 0.1, # 垂直平移范围    
    featurewise_center = True, 
    featurewise_std_normalization = True)

train_generator = train_datagen.flow_from_dataframe(
	train, directory = os.path.join('../input/cassava-leaf-disease-classification/train_images'),
 	subset = "training", x_col = "image_id", y_col = "label",
    target_size = (TARGET_SIZE, TARGET_SIZE), 
    batch_size = BATCH_SIZE, class_mode = "sparse", shuffle= True)

validation_datagen = ImageDataGenerator(validation_split = 0.2)

validation_generator = validation_datagen.flow_from_dataframe(
	train, directory = os.path.join('../input/cassava-leaf-disease-classification/train_images'),
    subset = "validation", x_col = "image_id", y_col = "label",
    target_size = (TARGET_SIZE, TARGET_SIZE),
    batch_size = BATCH_SIZE, class_mode = "sparse")


# img_path = os.path.join('../input/cassava-leaf-disease-classification/train_images/1003442061.jpg')
# img = image.load_img(img_path, target_size = (TARGET_SIZE, TARGET_SIZE))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis = 0)
# img_tensor /= 255.
# plt.imshow(img_tensor[0])
# plt.show()

# generator = train_datagen.flow_from_dataframe(train.iloc[17:18],
#                          directory = os.path.join('../input/cassava-leaf-disease-classification/train_images'),
#                          x_col = "image_id",
#                          y_col = "label",
#                          target_size = (TARGET_SIZE, TARGET_SIZE),
#                          batch_size = BATCH_SIZE,
#                          class_mode = "sparse")

# aug_images = [generator[0][0][0]/255 for i in range(10)]
# fig, axes = plt.subplots(2, 5, figsize = (20, 10))
# axes = axes.flatten()
# for img, ax in zip(aug_images, axes):
#     ax.imshow(img)
# plt.tight_layout()
# plt.show()


######### 模型部分
conv_base = Xception(include_top=False,
					 input_tensor=None,
					 pooling=None, 
					 input_shape=(TARGET_SIZE, TARGET_SIZE, 3), 
					 classifier_activation='softmax')
out = conv_base.output					 
out = layers.GlobalAveragePooling2D()(out)
out = layers.Dense(5, activation = "softmax")(out)
net = models.Model(conv_base.input, out)
model = net
net.compile(optimizer = Adam(lr = 0.001),
            loss = "sparse_categorical_crossentropy",
            metrics = ["acc"])

# model_save = ModelCheckpoint('./Xception_best_weights2.h5', 
#                              save_best_only = True, 
#                              save_weights_only = True,
#                              monitor = 'val_loss', 
#                              mode = 'min', verbose = 1)
# early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
#                            patience = 5, mode = 'min', verbose = 1,
#                            restore_best_weights = True)
# reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.3, 
#                               patience = 2, min_delta = 0.001, 
#                               mode = 'min', verbose = 1) #reduced learning rate


# history = model.fit(
#     train_generator,
#     steps_per_epoch = STEPS_PER_EPOCH,
#     epochs = EPOCHS,
#     validation_data = validation_generator,
#     validation_steps = VALIDATION_STEPS,
#     callbacks = [model_save, early_stop, reduce_lr])
# model.save('./Xception_best_weights.h5')
# model = keras.models.load_model('../input/xception-best-weights/Xception_best_weights.h5')
# preds = []

# for image_id in submission_file.image_id:
#     image = Image.open(os.path.join(f'../input/cassava-leaf-disease-classification/test_images/{image_id}'))
#     image = image.resize((TARGET_SIZE, TARGET_SIZE))
#     image = np.expand_dims(image, axis = 0)
#     preds.append(np.argmax(model.predict(image)))

# submission_file['label'] = preds
# submission_file

# submission_file.to_csv('submission.csv', index = False)
