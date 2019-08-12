import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as k
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

tf.logging.set_verbosity(tf.logging.ERROR)
start = time.perf_counter()

SEED_RANDOM = 1337

DATA_FOLDER = 'data'
TRAIN_DATA_DIR, TEST_DATA_DIR = f'{DATA_FOLDER}/training/', f'{DATA_FOLDER}/testing/'
IMG_WIDTH, IMG_HEIGHT = 50, 50

NUM_EPOCHS = 40
BATCH_SZ = 16

MODEL_SAVE_PATH = 'models/model.h5'

print(f'Using GPU: {k.tensorflow_backend._get_available_gpus()}')
np.random.seed(SEED_RANDOM)

train_datagen = ImageDataGenerator(rescale=1./255., validation_split=0.25)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SZ,
    color_mode='grayscale',
    seed=SEED_RANDOM,
    shuffle=True,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SZ,
    color_mode='grayscale',
    seed=SEED_RANDOM,
    shuffle=True,
    class_mode='categorical'
)

num_classes = train_generator.num_classes

print("\n##\n")

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(40))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

model.summary()

print("\n##\n")

# this is the augmentation configuration we will use for training
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size,
    epochs=NUM_EPOCHS,
    verbose=2,
    max_queue_size=10000,
)

print(f"\n## Treinamento finalizado em {time.perf_counter() - start:.0f}s ##\n")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save_weights(MODEL_SAVE_PATH)
