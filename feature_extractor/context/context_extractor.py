import tensorflow as tf
from tensorflow.keras.applications import VGG16
import numpy as np

class ContextExtractor:
    def __init__(self):
        # Use the first six layers of vgg16
        convbase = VGG16(weights='imagenet',
                         include_top=True)
        layers = convbase.layers[:21]
        self.model = tf.keras.models.Sequential()

        for layer in layers:
            self.model.add(layer)

    def __call__(self, data):
        # data is a RGB numpy array represents an image
        # Should have size (224,224,3)
        # Return the feature vector of shape (4096,)
        assert data.shape == (224,224,3)
        data = np.array([data])
        return self.model.predict(data)[0]