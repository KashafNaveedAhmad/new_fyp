
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Create dummy dataset
X = np.random.rand(100, 224, 224, 3)
y = np.random.randint(0, 4, 100)

y = tf.keras.utils.to_categorical(y, 4)

# Build simple CNN model
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train (very small, just for demo)
model.fit(X, y, epochs=2)

# Save model
model.save("model.h5")

print("Model saved as model.h5")