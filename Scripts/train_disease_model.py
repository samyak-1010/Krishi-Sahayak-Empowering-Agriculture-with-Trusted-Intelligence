import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks

# ==============================
# Project Root (repo root)
# ==============================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ==============================
# Paths (all relative to repo)
# ==============================
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
PLANT_DIR = os.path.join(DATASET_DIR, "PlantVillage")  # Your PlantVillage folder with class subfolders
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==============================
# Constants
# ==============================
IMG_SIZE = (256, 256)
BATCH = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 10

# ==============================
# Data Generators (with validation_split)
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2],
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    PLANT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='training'  # Use as training data
)

val_generator = train_datagen.flow_from_directory(
    PLANT_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation'  # Use as validation data
)

num_classes = train_generator.num_classes
print("Num classes:", num_classes)
print("Classes mapping:", train_generator.class_indices)

# ==============================
# Model (EfficientNetB4)
# ==============================
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
    pooling="avg"
)
base.trainable = False  # Freeze backbone initially
x = layers.Dense(512, activation="relu")(base.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs=base.input, outputs=outputs)

# ==============================
# Optimizer + Compile
# ==============================
initial_learning_rate = 1e-3
lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True
)
optimizer = optimizers.Adam(learning_rate=lr_schedule)
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
print(model.summary())

# ==============================
# Callbacks
# ==============================
checkpoint_path = os.path.join(MODEL_DIR, "img_model_best.h5")
callbacks_list = [
    callbacks.ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor="val_auc",
        mode="max"
    ),
    callbacks.EarlyStopping(
        monitor="val_auc",
        patience=6,
        restore_best_weights=True
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]

# ==============================
# Training
# ==============================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1
)

# ==============================
# Fine-tuning
# ==============================
base.trainable = True
fine_tune_at = 150
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)
history_fine = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    verbose=1
)

# ==============================
# Save Model + Label Map
# ==============================
final_model_path = os.path.join(MODEL_DIR, "img_model_final.h5")
model.save(final_model_path)
print("✅ Saved model to:", final_model_path)
label_map_path = os.path.join(MODEL_DIR, "label_map.json")
with open(label_map_path, "w") as f:
    json.dump(train_generator.class_indices, f)
print("✅ Saved label map to:", label_map_path)
