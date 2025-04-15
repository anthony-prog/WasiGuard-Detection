import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

# ----------------------------------------
# Parámetros
# ----------------------------------------
IMG_SIZE = 256
NUM_CLASSES = 10
BATCH_SIZE = 4
EPOCHS = 10

TRAIN_PATH = 'train'
VAL_PATH = 'val'

# ----------------------------------------
# Cargar imágenes y máscaras
# ----------------------------------------
def load_images_and_masks(path_org, path_label, size, num_classes):
    imgs = []
    masks = []

    # Leer imágenes originales
    valid_files = [f for f in os.listdir(path_org) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔍 Archivos encontrados en {path_org}: {len(valid_files)}")

    matched_files = []
    for f in valid_files:
        base_name = os.path.splitext(f)[0]
        label_filename = f"{base_name}_lab.png"
        label_path = os.path.join(path_label, label_filename)
        if os.path.exists(label_path):
            matched_files.append((f, label_filename))

    print(f"🔗 Archivos con máscara correspondiente: {len(matched_files)}")

    for img_file, label_file in matched_files:
        img = cv2.imread(os.path.join(path_org, img_file))
        img = cv2.resize(img, (size, size))
        img = img.astype('float32') / 255.0

        label = cv2.imread(os.path.join(path_label, label_file), cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)
        label = to_categorical(label, num_classes)

        imgs.append(img)
        masks.append(label)

    print(f"✅ Cargadas {len(imgs)} imágenes válidas desde {path_org}")
    return np.array(imgs), np.array(masks)

# ----------------------------------------
# Definir modelo U-Net básico
# ----------------------------------------
def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)

    # Decoder
    u1 = UpSampling2D()(c3)
    m1 = concatenate([u1, c2])
    c4 = Conv2D(128, 3, activation='relu', padding='same')(m1)

    u2 = UpSampling2D()(c4)
    m2 = concatenate([u2, c1])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(m2)

    outputs = Conv2D(num_classes, 1, activation='softmax')(c5)

    return Model(inputs, outputs)

# ----------------------------------------
# Entrenamiento
# ----------------------------------------
def main():
    print("📥 Cargando datos...")
    X_train, y_train = load_images_and_masks(
        os.path.join(TRAIN_PATH, 'train-org-img'),
        os.path.join(TRAIN_PATH, 'train-label-img'),
        IMG_SIZE,
        NUM_CLASSES
    )

    X_val, y_val = load_images_and_masks(
        os.path.join(VAL_PATH, 'val-org-img'),
        os.path.join(VAL_PATH, 'val-label-img'),
        IMG_SIZE,
        NUM_CLASSES
    )

    print("🔧 Compilando modelo U-Net...")
    model = unet_model()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    print("🚀 Entrenando modelo...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )

    model_filename = 'floodnet_unet_model.h5'
    model.save(model_filename)
    print(f"✅ Modelo guardado como '{model_filename}'")

if __name__ == '__main__':
    main()
