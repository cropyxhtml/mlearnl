import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
class FashionChecker:
    def __init__(self):
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def create_model(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        plt.figure
        plt.imgshow(train_images[10])
        plt.colorbar()
        plt.grid(Fasle)
        plt.show()

        # train_images = train_images / 255.0
        # train_labels = train_labels / 255.0
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss ='separse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(train_images,train_labels,epochs=5)
