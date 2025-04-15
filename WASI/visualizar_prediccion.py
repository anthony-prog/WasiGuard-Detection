import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tkinter import Tk, filedialog

# --- Configuración ---
IMG_SIZE = 256
NUM_CLASSES = 10
MODEL_PATH = "floodnet_unet_model.h5"

# --- Selección de imagen ---
Tk().withdraw()  # Oculta la ventana principal de Tk
image_path = filedialog.askopenfilename(
    title="Selecciona una imagen para predecir",
    filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
)

if not image_path:
    print("❌ No se seleccionó ninguna imagen.")
    exit()

# --- Cargar modelo ---
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente")

# --- Cargar y preprocesar imagen ---
img = cv2.imread(image_path)
original_img = cv2.cvtColor(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)
img_input = original_img.astype('float32') / 255.0
img_input = np.expand_dims(img_input, axis=0)

# --- Predicción ---
pred = model.predict(img_input)[0]
mask = np.argmax(pred, axis=-1)

# --- Colorear máscara ---
def color_mask(mask):
    colors = [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
        (0, 255, 255), (139, 69, 19), (255, 192, 203), (128, 0, 128), (124, 252, 0)
    ]
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(colors):
        color_img[mask == cls] = color
    return color_img

colored_mask = color_mask(mask)

# --- Mostrar resultados ---
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Imagen Original")
plt.imshow(original_img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Máscara Predicha")
plt.imshow(colored_mask)
plt.axis('off')

plt.tight_layout()
plt.show()
