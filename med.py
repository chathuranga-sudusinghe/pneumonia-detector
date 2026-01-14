import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

train_ds=tf.keras.utils.image_dataset_from_directory(

	"/home/saviru/med/archive(4)/chest_xray/train",
	image_size=(128,128),
	batch_size=32,	
	seed=42,
	validation_split=0.2,
	subset='training'
)
validation_ds=tf.keras.utils.image_dataset_from_directory(

	"/home/saviru/med/archive(4)/chest_xray/train",
	image_size=(128,128),
	batch_size=32,
	seed=42,
	validation_split=0.2,
	subset='validation'
)
test_ds=tf.keras.utils.image_dataset_from_directory(

	"/home/saviru/med/archive(4)/chest_xray/test",
	image_size=(128,128),
	batch_size=32,
	shuffle=False
)

print(train_ds.class_names)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

#build the model
augmentation=tf.keras.Sequential([

	tf.keras.layers.RandomFlip('horizontal'),
	tf.keras.layers.RandomZoom(0.1),
	tf.keras.layers.RandomRotation(0.1)

])
model=tf.keras.Sequential([

    tf.keras.layers.Input(shape=(128,128,3)),
	augmentation,
	tf.keras.layers.Rescaling(1./255),
	
	tf.keras.layers.Conv2D(32,(3,3),activation='relu'),	
	tf.keras.layers.MaxPooling2D(),

	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),	
	tf.keras.layers.MaxPooling2D(),
	
	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),	
	tf.keras.layers.MaxPooling2D(),
	
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Dense(128,activation='relu'),
	tf.keras.layers.Dense(64,activation='relu'),
	tf.keras.layers.Dense(1,activation='sigmoid')
])
#compile the model
model.compile(

	loss=tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=['accuracy']
)

#balance the classes
class_weights=compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1]),
    y=np.concatenate([y.numpy() for _, y in train_ds])
)

class_weight={0: class_weights[0], 1: class_weights[1]}

#train the model
epoch_number=10
model.fit(train_ds,validation_data=validation_ds,epochs=epoch_number,class_weight=class_weight)

joblib.dump(test_ds.class_names,"class_names.pkl")
joblib.dump(0.60,"decision_threshold.pkl")

#evaluate the model
tess_loss,test_accuracy=model.evaluate(test_ds)
print("test accuracy",test_accuracy)

img_path='pne3.jpg'
img=image.load_img(img_path,target_size=(128,128))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)

plt.imshow(img)
plt.show()

pred=model.predict(img_array)

print(pred[0][0])

if pred[0][0]>=0.60:
 print('pneumonia')
else:
 print('normal')

y_true=[]
y_pred=[]

for images,labels in test_ds:
    preds=model.predict(images)
    preds=(preds >= 0.60).astype(int).flatten()  # threshold
    y_pred.extend(preds)
    y_true.extend(labels.numpy())

#Classification Report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=test_ds.class_names
))
#confusion matrix
cm=confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_ds.class_names,
    yticklabels=test_ds.class_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

model.save('pneumonia_model.h5')
