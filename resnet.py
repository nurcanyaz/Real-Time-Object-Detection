##################################################
# Real Time Object Detection with ResNet-50 Model
##################################################

import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras

# ResNet-50 modelini yükleme
model = ResNet50(weights='imagenet')

cap = cv2.VideoCapture(0)

while(True):
  ret, frame = cap.read()
  if not ret:
    break

  img = cv2.resize(frame, (224, 224))
  img = np.expand_dims(img, axis=0)
 # img = image.img_to_array(img)
  img = preprocess_input(img)

  # Görüntüyü model ile tahmin etme
  preds = model.predict(img)

  # Tahminlerden sınıf etiketlerini alma
  label = decode_predictions(preds, top=1)[0][0]
  #class_name, class_description, _=label
  label_name, label_cinfidence = label[1], label[2]

  # Tahminleri ekrana yazdırma
  cv2.putText(frame, f"{label_name} ({label_cinfidence*100:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

  cv2.imshow("Real Time Object Detection", frame)

  # 'q' tuşuna basıldığında çıkış
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()