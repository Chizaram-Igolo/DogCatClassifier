# 1. Examine and understand data
# 2. Build an input pipeline
# 3. Build the model
# 4. Train the model
# 5. Test the model
# 6. Improve the model and repeat the process

# Import TensorFlow and other libraries
import math

import matplotlib.pyplot as plt
import os
import cv2
from glob import glob
import tensorflow as tf
import pathlib

# Download and explore the dataset
data_dir = pathlib.Path("data/")
print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# dogs = list(data_dir.glob('dog/*'))

# img = cv2.imread(str(dogs[1]))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()

# Load using keras.preprocessing

# Create a dataset
# Define some parameters for the loader:
batch_size = 32
img_height = 128
img_width = 128

# It's good practice to use a validation split when developing your model. Let's use 80% of the
# images for training, and 20% for validation.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = val_ds.class_names
plt.figure(figsize=(10, 10))

# take a single batch
for images, labels in train_ds.take(1):
    for i in range(32):
        ax = plt.subplot(6, 6, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()

# Configure the dataset for performance using buffered prefetching and caching.

# Dataset.cache() keeps the images in memory after they're loaded off disk during first epoch.
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

num_classes = len(class_names)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3),
                           padding='same',
                           input_shape=(img_height, img_width, 3),
                           activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

print(model.summary())

epochs = 10
history = model.fit(
    normalized_ds,
    validation_data=normalized_val_ds,
    epochs=epochs
)

test_dogs = list(data_dir.glob("test_data/*"))

test_data_dir = pathlib.Path("test_data/")

test_imgs = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    test_imgs.extend(glob(os.path.join(test_data_dir, ext)))

for i in range(len(test_imgs)):
    test_image = cv2.imread(str(test_imgs[i]))
    test_image_resized = cv2.resize(test_image, (img_height, img_width), cv2.INTER_LINEAR) / 255
    test_pred = model.predict(test_image_resized.reshape(1, img_height, img_width, 3))
    print("{:.2f}% Dog, {:.2f}% Cat".format(test_pred[0][0] * 100, test_pred[0][1] * 100))

plt.figure(figsize=(20, 20))
dimensions = math.ceil(math.sqrt(len(test_imgs)))
rows, columns = dimensions, dimensions

# take a single batch
for i in range(len(test_imgs)):
    print(test_imgs[i])
    img = cv2.imread(str(test_imgs[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(rows, columns, i + 1)
    plt.imshow(img)
    plt.title(test_imgs[i])
    plt.axis("off")

plt.show()

params = ['loss', 'accuracy']

plt.figure(figsize=(14, 6))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.xlabel('Epoch', )
    plt.ylabel('Loss')
    plt.plot(history.epoch, history.history[params[i]], label=f'{params[i]}')
    plt.plot(history.epoch, history.history[f'val_{params[i]}'], label=f'val_{params[i]}')
    plt.legend()

plt.show()