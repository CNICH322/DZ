import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train_vgg = np.repeat(x_train, 3, axis=-1)
x_test_vgg = np.repeat(x_test, 3, axis=-1)
x_train_vgg = tf.image.resize(x_train_vgg, (32, 32))
x_test_vgg = tf.image.resize(x_test_vgg, (32, 32))

print("Train:", x_train_vgg.shape)
print("Test:", x_test_vgg.shape)

y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
conv_base.trainable = False

model_vgg = Sequential([conv_base, Flatten(), Dense(256, activation="relu"), Dense(10, activation="softmax")])

model_vgg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_vgg.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

print("\nПЕРШИЙ ЕТАП: заморожена VGG16\n")
history = model_vgg.fit(x_train_vgg, y_train_cat, epochs=15, batch_size=128, validation_split=0.2, callbacks=[early_stop])

print("\nДРУГИЙ ЕТАП: fine-tuning\n")
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == "block5_conv1":
        set_trainable = True
    layer.trainable = set_trainable

model_vgg.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history_fine = model_vgg.fit(x_train_vgg, y_train_cat, epochs=5, batch_size=128, validation_split=0.2, callbacks=[early_stop])

combined_history = {}
for key in history.history.keys():
    combined_history[key] = history.history[key] + history_fine.history[key]

os.makedirs("models", exist_ok=True)
os.makedirs("histories", exist_ok=True)

model_vgg.save("models/vgg16_model.keras")
print("\nМодель збережено: models/vgg16_model.keras")

with open("histories/vgg16_history.json", "w") as file:
    json.dump(combined_history, file)
print("Історію збережено: histories/vgg16_history.json")

epochs = range(1, len(combined_history["loss"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, combined_history["loss"], label="Training loss")
plt.plot(epochs, combined_history["val_loss"], label="Validation loss")
plt.title("Втрати VGG16")
plt.xlabel("Епоха")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs, combined_history["accuracy"], label="Training accuracy")
plt.plot(epochs, combined_history["val_accuracy"], label="Validation accuracy")
plt.title("Точність VGG16")
plt.xlabel("Епоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

print("\nОЦІНКА МОДЕЛІ\n")
test_loss, test_acc = model_vgg.evaluate(x_test_vgg, y_test_cat, verbose=1)
print("Точність VGG16:", test_acc)
print("Втрата VGG16:", test_loss)
