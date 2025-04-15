import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

# --- Configuración ---
IMG_SIZE = 256
NUM_CLASSES = 10
MODEL_PATH = "floodnet_unet_model.h5"
CLASES_HUAICO = [1, 3, 5]  # Building-flooded, Road-flooded, Water

# --- Seleccionar imagen ---
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Selecciona una imagen para analizar",
    filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
)
if not image_path:
    print("❌ No se seleccionó imagen.")
    exit()

# --- Cargar modelo ---
model = load_model(MODEL_PATH)
print("✅ Modelo cargado.")

# --- Procesar imagen ---
img = cv2.imread(image_path)
if img is None:
    print(f"❌ Error leyendo la imagen: {image_path}")
    exit()

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_input = img_rgb.astype('float32') / 255.0
img_input = np.expand_dims(img_input, axis=0)

# --- Predicción ---
pred = model.predict(img_input)[0]
mask = np.argmax(pred, axis=-1)  # (256, 256)

# --- Cálculo de porcentaje de huaico ---
total_pixels = mask.size
huaico_pixels = np.isin(mask, CLASES_HUAICO).sum()
porcentaje = (huaico_pixels / total_pixels) * 100

print(f"🧾 Porcentaje de imagen con evidencia de huaico/inundación: {porcentaje:.2f}%")

# --- Mostrar resultado visual y alerta ---
if porcentaje < 5:
    mensaje = "✅ No hay huaico detectado"
elif porcentaje < 20:
    mensaje = "⚠️ Huaico detectado"
else:
    mensaje = "🚨 Huaico severo detectado"

# Mostrar imagen y resultado
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title(mensaje , fontsize=14)
plt.axis('off')
plt.show()
