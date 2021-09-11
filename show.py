import matplotlib.pyplot as plt
import numpy as np

def plot_images(images):
    plt.figure(figsize=(32, 32))

    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        image = images[i, :, :, :]
        image = (image + 1) / 2.0  # Rescale to [0, 255]
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('GAN_cat1.png')

from tensorflow.keras.models import load_model

G = load_model('g.h5')

new_images = G.predict(np.random.normal(0, 1, (25, 100)))
plot_images(new_images)
