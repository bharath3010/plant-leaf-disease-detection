import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def load_model():
    model = tf.keras.models.load_model('models/leaf_disease_model.h5')
    return model

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension
    return img_array / 255.0

def main(image_path):
    model = load_model()
    img = prepare_image(image_path)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    
    print(f"Predicted class: {predicted_class}")

if __name__ == '__main__':
    image_path = sys.argv[1]
    main(image_path)


