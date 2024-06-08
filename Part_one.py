import json

import h5py
import tensorflow as tf
from keras import Sequential
from keras.src.datasets import cifar10, mnist
from tensorflow.keras.layers import Dense, Reshape, Flatten, BatchNormalization, LeakyReLU, Conv2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import keras
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


def visualize_results(generator, latent_dim, n_samples=10):
    """
    Визуализирует результаты работы генератора.

    :param generator: Модель генератора
    :param latent_dim: Размерность латентного пространства
    :param n_samples: Количество сгенерированных изображений для отображения
    """
    # Генерация случайного латентного вектора
    random_latent_vectors = np.random.normal(0, 1, (n_samples, latent_dim))

    # Генерация изображений
    generated_images = generator.predict(random_latent_vectors)

    # Нормализация изображений
    generated_images = (generated_images + 1) / 2.0  # Преобразование от [-1, 1] к [0, 1]

    # Определение размера сетки для отображения изображений
    grid_size = int(np.sqrt(n_samples))

    # Создание фигуры
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for img, ax in zip(generated_images, axes):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# endregion

# region Модели
# Построение генератора
def build_generator(latent_dim_inner):
    model = Sequential()

    # Входной слой: плотный (Dense) слой
    model.add(Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim_inner,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Преобразование в 3D-форму (7, 7, 256)
    model.add(Reshape((7, 7, 256)))  # Начальная форма 7x7x256

    # Первый слой транспонированной свертки (Conv2DTranspose)
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Второй слой транспонированной свертки (увеличение размера до 14x14)
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Третий слой транспонированной свертки (увеличение размера до 28x28)
    model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Выходной слой (одноканальное изображение 28x28)
    model.add(Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model


# Построение дискриминатора
def build_discriminator(image_shape):
    model = Sequential()

    # Первый сверточный слой
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=image_shape))
    model.add(LeakyReLU())

    # Второй сверточный слой
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())

    # Третий сверточный слой
    model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())

    # Выравнивание (Flatten) и плотный (Dense) слой
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
            # Сохранение генератора
            generator_inner.save(f'generator-epoch-{epoch}.h5')

            # Сохранение дискриминатора
            discriminator_inner.save(f'discriminator-epoch-{epoch}.h5')

            # Сохранение всей модели GAN
            gan_inner.save(f'gan-epoch-{epoch}.h5')

    return gan_inner


# endregion


def train_model():
    print("Начинаю загрузку датасета MNIST")
    data = load_mnist()
    print("Датасет MNIST загружен")

    # Создание моделей
    # Основные параметры
    latent_dim = 100
    img_shape = (28, 28, 1)
    epochs = 1
    batch_size = 16
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
    generator.save('generator_result.h5')

    # Сохранение дискриминатора
    discriminator.save('discriminator_result.h5')

    # Сохранение всей модели GAN
    gan.save('gan_result.h5')
    result_training_gan.save('result_training_gan_result.h5')


def load_and_compile_model(filepath):
    remove_groups_param_from_h5(filepath)
    model = keras.models.load_model(filepath)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def check_result(latent_dim_inner=100):
    start_epoch = 15
    generator = load_and_compile_model(
        rf"D:\я у мамы программист\3 курс 2 семестр КЗ\lab_7\pythonProject\generator-epoch-{start_epoch}.h5")

    visualize_results(generator, latent_dim_inner, 100)


def remove_groups_param_from_h5(filepath_inner):
    with h5py.File(filepath_inner, 'r+') as f:
        model_config = f.attrs['model_config']
        model_config = json.loads(model_config)

        for layer in model_config['config']['layers']:
            if 'groups' in layer['config']:
                del layer['config']['groups']

        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')


if __name__ == "__main__":
    check_result()
