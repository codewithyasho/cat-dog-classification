
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

# --- Data Loading ---
image_size = (150, 150)
batch_size = 32

train_dataset = keras.utils.image_dataset_from_directory(
    directory='cat-dog-data/train',
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size
)

validation_dataset = keras.utils.image_dataset_from_directory(
    directory='cat-dog-data/val',  # Separate validation folder
    labels='inferred',
    label_mode='int',
    image_size=image_size,
    batch_size=batch_size
)

# --- Normalization ---
def process(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_dataset = train_dataset.map(process).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(process).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- Data Augmentation ---
data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# --- Model Architecture ---
model = Sequential()
model.add(data_augmentation)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.summary()

# --- Compile Model ---
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_model.h5", save_best_only=True)
]

# --- Train Model ---
model_history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# --- Save Model & History ---
import joblib
joblib.dump(model_history.history, "model_history.pkl")
model.save("final_model.h5")
