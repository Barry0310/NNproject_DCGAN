# -*- coding: utf-8 -*-
"""# Step 1: Load dataset"""

import numpy as np
import os
import matplotlib.pyplot as plt

def plot_images(images, num):
    plt.figure(figsize=(32, 32))

    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        image = images[i, :, :, :]
        image = (image + 1) / 2.0  # Rescale to [0, 255]
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('GAN_pic/' + str(num) + '.png')
    plt.close()


# save model
from tensorflow.keras.models import save_model

def model_save(GAN, G, D):
    save_model(G, 'g.h5')

    # for resume training
    D.trainable = True
    save_model(D, 'd.h5')

    D.trainable = False
    save_model(GAN, 'gan.h5')


file_dir = 'cats/'
file = os.listdir(file_dir)
dataset = []
for i in range(0, 10000):
    dataset.append(plt.imread(file_dir+file[i]))
trainset = np.array(dataset)

# Rescale 0 to 1
trainset = (trainset.astype(np.float32) - 127.5) / 127.5

in_shape = dataset[0].shape




"""# Step 2: Define generator"""

from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Define generator
G = Sequential()
# foundation for 8x8 image
G.add(Dense(256 * 8 * 8, input_dim=100))
G.add(LeakyReLU(alpha=0.2))

G.add(Reshape((8, 8, 256)))

# upsample to 16x16
G.add(Conv2DTranspose(256, kernel_size=4, padding='same'))
G.add(LeakyReLU(alpha=0.2))

# upsample to 32x32
G.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
G.add(LeakyReLU(alpha=0.2))

# upsample to 64x64
G.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
G.add(LeakyReLU(alpha=0.2))

# output layer
G.add(Conv2DTranspose(3, kernel_size=4, strides=2, activation='tanh', padding='same'))

G.summary()

"""# Step 3: Define discriminator"""

# Define discriminator
D = Sequential()

# normal
D.add(Conv2D(64, kernel_size=4, padding='same', strides=2, input_shape=in_shape))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(128, kernel_size=4, padding='same', strides=2))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(256, kernel_size=4, padding='same', strides=2))
D.add(LeakyReLU(alpha=0.2))

# downsample
D.add(Conv2D(512, kernel_size=4, padding='same', strides=2))
D.add(LeakyReLU(alpha=0.2))


# classifier
D.add(Flatten())
D.add(Dropout(0.5))
D.add(Dense(1, activation='sigmoid'))

# compile model
D.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
D.summary()

"""# Step 4: Define GAN (concatenate generator and discriminator)"""

# Define GAN
def define_gan(g_model, d_model):

    # GAN is training Generator by the loss of Disciminator, make weights in the discriminator not trainable
    d_model.trainable = False

    model = Sequential()

    # concatenate generator and discriminator
    model.add(g_model)
    model.add(d_model)

    return model

# build GAN
GAN = define_gan(G, D)

# compile model
GAN.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

GAN.summary()

"""# Step 5: Train model"""

import math

# Configs
max_epoch = 200
batch_size = 32
max_fake_acc = 0

# Train GAN
for epoch in range(max_epoch):

    for i in range(math.ceil(len(trainset) / batch_size)):

        # Update discriminator by real samples
        r_images = trainset[i*batch_size:(i+1)*batch_size]
        d_loss_r, _ = D.train_on_batch(r_images, np.ones((len(r_images), 1)))

        # Update discriminator by fake samples
        f_images = G.predict(np.random.normal(0, 1, (batch_size, 100))) # generate fake images
        d_loss_f, _ = D.train_on_batch(f_images, np.zeros((len(f_images), 1)))

        d_loss = (d_loss_r + d_loss_f)/2

        # Update generator
        g_loss = GAN.train_on_batch(np.random.normal(0, 1, (batch_size, 100)), np.ones((batch_size, 1)))

        # Print training progress
        print(f'[Epoch {epoch+1}, {min((i+1)*batch_size, len(trainset))}/{len(trainset)}] D_loss: {d_loss:0.4f}, G_loss: {g_loss:0.4f}')

    # Print validation result
    # evaluate discriminator on real examples
    _, acc_real = D.evaluate(trainset, np.ones((len(trainset), 1)), verbose=0)

    # evaluate discriminator on fake examples
    f_images = G.predict(np.random.normal(0, 1, (len(trainset), 100)))
    _, acc_fake = D.evaluate(f_images, np.ones((len(trainset), 1)), verbose=0)

    # summarize discriminator performance
    print(f'[Epoch {epoch+1}] Accuracy real: {acc_real*100}, fake: {acc_fake*100}')
    plot_images(f_images[:25], epoch+1)
    if(max_fake_acc<acc_fake):
        print(max_fake_acc, '->', acc_fake, 'save model')
        max_fake_acc = acc_fake
        model_save(GAN, G, D)
