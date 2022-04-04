import numpy as np
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image


def load_process(path: str) -> np.ndarray:
    img = image.load_img(path, target_size=(224, 224))
    xs = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    return xs


# path = '/Users/kriangsakthuiprakhon/Documents/seniorProject/image/holiday_test/100400.jpg'
#
# if __name__ == '__main__':
#     load_process(path)
