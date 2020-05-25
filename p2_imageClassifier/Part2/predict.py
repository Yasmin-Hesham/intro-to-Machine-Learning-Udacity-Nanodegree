# Needed Libraries
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import argparse
import json

image_path="./test_images/"
loaded_model="./better_model.h5"
model_path="./models/better_model.h5"

# Using parser for user's input
parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", help="./test_images/", required=False, default="./test_images/wild_pansy.jpg")
parser.add_argument("-m","--model_path", help="model_path", required=False,default="./models/better_model.h5")
parser.add_argument("-k","--top_k", help="top k probs of the image",required=False, default=5)
parser.add_argument("-c","--category_names",help="classes",required=False,default="./label_map.json")
args = vars(parser.parse_args())

image_flower= args['image']
saved_model = args['model_path']
top_k = int(args['top_k'])
category_names = args['category_names']
image_size = 224

# Function "process_image"
def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    imageFinal = image.numpy()
    return imageFinal

# function "predict"
def predict(image,model,top_k):
   # top_k_tensor = tf.convert_to_tensor(top_k)
    imageLoad = Image.open(image)
    imageNp = np.asarray(imageLoad)
    imageProcess = process_image(imageNp)
    imageFinal = np.expand_dims(imageProcess,axis=0)
    
    probs = model.predict(imageFinal)
    prob_predictions= probs[0].tolist()
    probs_final, classes = tf.math.top_k(prob_predictions, k=top_k)
    
    probs_list = probs_final.numpy().tolist()
    index_shifted = classes.numpy()+1
    index = index_shifted.tolist()
    
    return probs_list, index

# checking inputs
if model_path != None:
     model = tf.keras.models.load_model(args['model_path'], custom_objects={'KerasLayer':hub.KerasLayer})
     model.summary()
else:
    print("Invalid path")
    

if category_names != None:
     with open(category_names, 'r') as f:
            class_names = json.load(f)
else:
    print("Invalid path")

    
# get top_K probabilities and index
topK_probs, topK_index = predict(image_flower, model, top_k)

print(type(topK_index))
print("Output of TOP Probabilities and Indices of Flowers:\n")
for i in range(len(topK_index)):
    print('\u2022',class_names.get(str(topK_index[i])))
    print('Probabilties:', topK_probs[i])
    print('Classes Keys:', topK_index[i])

#good reference: https://www.youtube.com/watch?v=q94B9n_2nf0