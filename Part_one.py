import tensorflow as tf
import concurrent.futures
from keras import Sequential
from keras.src.datasets import cifar10, mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# region Загрузка и нормализация данных CIFAR-10
def load_cifar10():
    (train_images, _), (test_images, _) = cifar10.load_data()
    data = np.concatenate((train_images, test_images))
    data = data.astype('float32')
    data = (data - 127.5) / 127.5  # Нормализация изображений в диапазон [-1, 1]
    return data


def load_mnist():
    (train_images, _), (test_images, _) = mnist.load_data()
    data = np.concatenate((train_images, test_images))
    data = data.astype('float32')

    # MNIST images are grayscale with shape (28, 28), expand dims to (28, 28, 1)
    data = np.expand_dims(data, axis=-1)

    # Normalize images to range [-1, 1]
    data = (data - 127.5) / 127.5

    return data

# endregion

# region Визуализация
# Функция для сохранения изображений
def save_images(generator_inner, epoch, latent_dim_inner, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, latent_dim_inner))
    gen_images = generate_images(generator_inner, noise)
    gen_images = 0.5 * gen_images + 0.5  # Переход от [-1, 1] к [0, 1]

    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(gen_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"gan_generated_image_epoch_{epoch}.png")
    plt.close()


# Демонстрация результатов генерации
def display_generated_images(generator_inner, latent_dim_inner, examples=10):
    noise = np.random.normal(0, 1, (examples, latent_dim_inner))
    gen_images = generator_inner.predict(noise)
    gen_images = 0.5 * gen_images + 0.5  # Обратная нормализация изображений в диапазон [0, 1]
    fig, axs = plt.subplots(1, examples, figsize=(15, 15))
    for i in range(examples):
        axs[i].imshow(gen_images[i])
        axs[i].axis('off')
    plt.show()


# endregion

# region Модели
# Построение генератора
def build_generator(latent_dim_inner):
    model = Sequential()
    model.add(Dense(2 * 2 * 256, use_bias=False, input_shape=(latent_dim_inner,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((2, 2, 256)))  # Начальная форма 4x4x256
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


# Построение дискриминатора
def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU())
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model



# Построение и компиляция GAN
def build_gan(generator_inner, discriminator_inner, latent_dim_inner):
    discriminator_inner.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                                metrics=['accuracy'])
    discriminator_inner.trainable = True
    gan_input = tf.keras.Input(shape=(latent_dim_inner,))
    gan_output = discriminator_inner(generator_inner(gan_input))
    gan_inner = tf.keras.Model(gan_input, gan_output)
    gan_inner.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return gan_inner


# endregion

# region Обучение
# Функция для тренировки дискриминатора
def train_discriminator(discriminator_inner, real_images, fake_images, real_labels, fake_labels):
    d_loss_real = discriminator_inner.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator_inner.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    return d_loss


# Функция для тренировки генератора

def train_generator(gan_inner, noise, real_labels):
    g_loss = gan_inner.train_on_batch(noise, real_labels)
    return g_loss


# Предварительная компиляция функции генерации изображений
def generate_images(generator_inner, noise):
    return generator_inner(noise, training=False)


# Обучение GAN
def train_gan(gan_inner, generator_inner, discriminator_inner, data_inner, epochs_inner, batch_size_inner,
              latent_dim_inner, save_interval_inner):
    real = np.ones((batch_size_inner, 1))
    fake = np.zeros((batch_size_inner, 1))

    for epoch in range(epochs_inner):
        for _ in tqdm(range(len(data_inner) // batch_size_inner), desc=f"Обучение GAN {epoch}/{epochs_inner}"):
            idx = np.random.randint(0, data_inner.shape[0], batch_size_inner)
            images = data_inner[idx]

            noise = np.random.normal(0, 1, (batch_size_inner, latent_dim_inner))
            gen_images = generate_images(generator_inner, noise)

            d_loss = train_discriminator(discriminator_inner, images, gen_images, real, fake)

            noise = np.random.normal(0, 1, (batch_size_inner, latent_dim_inner))
            g_loss = train_generator(gan_inner, noise, real)

        if epoch % save_interval_inner == 0:
            save_images(generator_inner, epoch, latent_dim_inner)
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

    return gan_inner


# endregion


# Загрузка и нормализация данных
print("Начинаю загрузку датасета CIFAR-10")
data = load_cifar10()
print("Датасет CIFAR-10 загружен")

# print("Начинаю загрузку датасета MNIST")
# data = load_mnist()
# print("Датасет MNIST загружен")

# Создание моделей
# Основные параметры
latent_dim = 100
img_shape = (32, 32, 3)
epochs = 1
batch_size = 32
save_interval = 1
print("Создаю модель")
generator = build_generator(latent_dim)
discriminator = build_discriminator(img_shape)
gan = build_gan(generator, discriminator, latent_dim)

# Обучение модели
print("Начинаю обучение модели")
result_training_gan = train_gan(gan, generator, discriminator, data, epochs, batch_size, latent_dim, save_interval)

display_generated_images(generator, latent_dim)

# Сохранение генератора
generator.save('generator.h5')

# Сохранение дискриминатора
discriminator.save('discriminator.h5')

# Сохранение всей модели GAN
gan.save('gan.h5')
result_training_gan.save('result_training_gan.h5')
