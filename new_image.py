import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import random

positive_pairs_dir = 'positive_pairs'
negative_pairs_dir = 'negative_pairs'

def get_negative_pairs():
    max_pairs = 1000
    pairs = []
    images = [os.path.join(negative_pairs_dir, f) for f in os.listdir(negative_pairs_dir) if f.endswith('.jpg')]
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append([images[i], images[j]])
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    return pairs

def get_positive_pairs():
    pairs = []
    max_pairs = 1000
    for subdir in os.listdir(positive_pairs_dir):
        subdir_path = os.path.join(positive_pairs_dir, subdir)
        if os.path.isdir(subdir_path):
            images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')]
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pairs.append([images[i], images[j]])
                    if len(pairs) >= max_pairs:
                        break
                if len(pairs) >= max_pairs:
                    break

    random.shuffle(pairs)
    return pairs     

def get_pairs():
    negative = get_negative_pairs()
    positive = get_positive_pairs()

    min_pairs = min(len(positive), len(negative))

    positive = positive[:min_pairs]
    negative = negative[:min_pairs]

    pairs = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)

    return pairs, labels

def preprocess_image(image, target_size):
    img = tf.keras.preprocessing.image.load_img(image, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def preprocess_pairs(pairs, target_size):
    processed_pairs = []
    for pair in pairs:
        img1 = preprocess_image(pair[0], target_size)
        img2 = preprocess_image(pair[1], target_size)

        processed_pairs.append([img1, img2])
    
    return processed_pairs

res_height = 256
res_width = 256
input_shape = (res_height, res_width, 3)

pairs, labels = get_pairs()

if len(pairs) == 0 or len(labels) == 0:
    raise ValueError('No data found')

pairs = preprocess_pairs(pairs, (res_height, res_width))
pairs = np.array(pairs)
labels = np.array(labels)

if len(pairs) == 0 or len(labels) == 0:
    raise ValueError('No data survived preprocessing')

train_pairs, val_pairs, train_labels, val_labels = train_test_split(pairs, labels, test_size=0.2, random_state=42)

train_pairs_1 = np.array([pair[0] for pair in train_pairs])
train_pairs_2 = np.array([pair[1] for pair in train_pairs])
val_pairs_1 = np.array([pair[0] for pair in val_pairs])
val_pairs_2 = np.array([pair[1] for pair in val_pairs])

if len(train_pairs_1) == 0 or len(train_pairs_2) == 0:
    raise ValueError('Empty set for training will not work')

if len(val_pairs_1) == 0 or len(val_pairs_2) == 0:
    raise ValueError('Empty set for validation will not work')

def create_base_network(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))  
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    return model

base_cnn = create_base_network(input_shape)

input_a = tf.keras.layers.Input(shape=input_shape)
input_b = tf.keras.layers.Input(shape=input_shape)

processed_a = base_cnn(input_a)
processed_b = base_cnn(input_b)

# Lambda layer with tf.keras.backend for absolute difference
dis = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]), output_shape=lambda x: x[0])([processed_a, processed_b])
op = tf.keras.layers.Dense(1, activation='sigmoid')(dis)

model = tf.keras.models.Model([input_a, input_b], op)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

full_history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

batch_size = len(train_pairs_1) // 10
quarter_size = len(train_pairs_1) // 4

for quarter in range(4):
    start_idx = quarter * quarter_size
    end_idx = start_idx + quarter_size
    
    train_pairs_1_quarter = train_pairs_1[start_idx:end_idx]
    train_pairs_2_quarter = train_pairs_2[start_idx:end_idx]
    train_labels_quarter = train_labels[start_idx:end_idx]

    val_pairs_1_quarter = val_pairs_1[start_idx:end_idx]
    val_pairs_2_quarter = val_pairs_2[start_idx:end_idx]
    val_labels_quarter = val_labels[start_idx:end_idx]

    if len(train_pairs_1_quarter) == 0 or len(val_pairs_1_quarter) == 0:
        break
    
    train_dataset = tf.data.Dataset.from_tensor_slices(((train_pairs_1_quarter, train_pairs_2_quarter), train_labels_quarter))
    val_dataset = tf.data.Dataset.from_tensor_slices(((val_pairs_1_quarter, val_pairs_2_quarter), val_labels_quarter))
    
    train_dataset = train_dataset.batch(32).repeat()
    val_dataset = val_dataset.batch(32).repeat()
    
    steps_per_epoch = len(train_pairs_1_quarter) // 32
    validation_steps = len(val_pairs_1_quarter) // 32
    
    history = model.fit(train_dataset, 
                        steps_per_epoch=steps_per_epoch, 
                        epochs=1, 
                        validation_data=val_dataset, 
                        validation_steps=validation_steps, 
                        callbacks=[early_stop])
    
    for key in full_history.keys():
        full_history[key].extend(history.history.get(key, []))
    
    model.evaluate(val_dataset, steps=validation_steps)

model.save('my_model.h5')

fig, axs = plt.subplots(2, 1, figsize=(10, 10))

axs[0].plot(full_history['accuracy'], label='Train Accuracy')
axs[0].plot(full_history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].legend(loc='lower right')

axs[1].plot(full_history['loss'], label='Train Loss')
axs[1].plot(full_history['val_loss'], label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].legend(loc='upper right')

plt.tight_layout()
plt.show()
