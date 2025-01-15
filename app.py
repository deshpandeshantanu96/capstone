import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) #We will use imagenet dataset from ResNET model, will convert all the images to (224,224,3)
model.trainable = False #As we dont want to train the ResNET model again

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_image_array = np.expand_dims(img_array, axis=0)
    preprocessed_image = preprocess_input(expanded_image_array)
    result = model.predict(preprocessed_image)
    normalized_result = result / norm(result)

    return normalized_result

#print(os.listdir('images'))

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

#print(filename[0:5])

feature_list = []

for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

print(np.array(feature_list).shape)

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

