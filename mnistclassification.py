# -*- coding: utf-8 -*-
"""MNISTclassification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DMdV7rO_THGHHWHurBlaScEYI_jS_Qv4
"""

import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

y_train = np.array([label % 2 for label in y_train])
y_test = np.array([label % 2 for label in y_test])

# entropy gradient function
def entropy_gradient(weights):
    return -weights * np.log(np.abs(weights) + 1e-8)

# improved neural network model with entropy gradient and constraints
def build_advanced_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(28*28,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-5)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_advanced_model()
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc:.4f}')

# entropy gradient update
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        new_weights = entropy_gradient(layer.get_weights()[0])
        model.get_layer(layer.name).set_weights([new_weights, layer.get_weights()[1]])

import matplotlib.pyplot as plt
import numpy as np

random_indices = np.random.choice(len(x_test), 10, replace=False)

plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    image = x_test[idx].reshape(28, 28)
    true_label = "Odd" if y_test[idx] == 1 else "Even"
    pred_label = "Odd" if model.predict(x_test[idx:idx+1]) >= 0.5 else "Even"

    plt.subplot(2, 5, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()