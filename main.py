import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as k, callbacks, applications
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from numpy import random

tf.logging.set_verbosity(tf.logging.ERROR)
start = time.perf_counter()

SEED_RANDOM = 1337

DATA_FOLDER = 'data'
TRAIN_DATA_DIR, TEST_DATA_DIR = f'{DATA_FOLDER}/training/', f'{DATA_FOLDER}/testing/'
IMG_WIDTH, IMG_HEIGHT = 50, 50

NUM_EPOCHS = 30
BATCH_SZ = 8

run_id = random.randint(low=1, high=1000)
MODEL_SAVE_PATH = f'models/({run_id}) {{val_acc:.2f}}-{{val_loss:.2f}} {{epoch:02d}}-{BATCH_SZ}.hdf5'

gpu_options = tf.GPUOptions(allow_growth=True)
print(f'Using GPU: {k.tensorflow_backend._get_available_gpus()}')

np.random.seed(SEED_RANDOM)

train_datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25,
                                   preprocessing_function=applications.densenet.preprocess_input)
test_datagen = ImageDataGenerator(rescale=1. / 255,
                                  preprocessing_function=applications.densenet.preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SZ,
    color_mode='rgb',
    seed=SEED_RANDOM,
    shuffle=True,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SZ,
    color_mode='rgb',
    seed=SEED_RANDOM,
    shuffle=True,
    class_mode='categorical'
)

num_classes = train_generator.num_classes

print("\n##\n")

tf_base = applications.densenet.DenseNet201(weights='imagenet',
                                            include_top=False,
                                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

for layer in tf_base.layers:
    layer.trainable = False

model = Sequential()
model.add(tf_base)

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))

model.compile('adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\n##\n")

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
save_callback = callbacks.ModelCheckpoint(MODEL_SAVE_PATH,
                                          monitor='val_acc',
                                          verbose=0,
                                          save_best_only=True,
                                          save_weights_only=False,
                                          mode='auto',
                                          period=1)

# this is the augmentation configuration we will use for training
history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_data=test_generator,
    validation_steps=test_generator.n // test_generator.batch_size,
    epochs=NUM_EPOCHS,
    verbose=2,
    max_queue_size=10000,
    callbacks=[save_callback, early_stopping]
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
