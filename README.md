# WasiGuard-Detection
# WasiGuard – Detection using AI 🧠🌊🏠

WasiGuard es un sistema inteligente de análisis visual que permite detectar zonas afectadas por huaicos e inundaciones mediante segmentación semántica en imágenes aéreas.

Este proyecto utiliza un modelo de tipo **U-Net**, entrenado con el dataset **FloodNet**, para identificar y clasificar automáticamente píxeles en categorías como edificios inundados, carreteras afectadas, agua visible, vegetación, entre otros.

## 📦 Contenido del repositorio

- `entrenar_floodnet.py`: Script para entrenar el modelo de segmentación U-Net.
- `visualizar_prediccion.py`: Permite cargar una imagen y visualizar la predicción del modelo.
- `detectar_huaico_porcentaje.py`: Analiza una imagen y calcula el porcentaje de área afectada por huaico/inundación.
- `/FLOODNET-SUPERVISED_V1.0/`: Estructura del dataset utilizado para entrenamiento y validación.
- `floodnet_unet_model.h5`: Modelo entrenado listo para ser usado.

## 🚨 ¿Qué hace?

- Clasifica cada píxel de una imagen aérea en una de 10 categorías.
- Calcula el porcentaje de afectación por huaico/inundación.
- Emite alertas visuales con niveles de riesgo (leve, moderado, severo).
- Puede usarse en aplicaciones móviles, drones, cámaras urbanas, etc.

## 🔧 Tecnologías usadas

- Python 3
- TensorFlow / Keras
- OpenCV
- Matplotlib
- FloodNet dataset

## 🧠 Objetivo

Proveer una solución accesible y automática para la **detección temprana de huaicos e inundaciones**, aplicable en zonas vulnerables como la cuenca del Rímac, Huaycán, Chosica, entre otras.

## 🛰️ Próximos pasos

- Exportar a TensorFlow Lite para integrarlo en aplicaciones móviles.
- Conectar con sistemas GIS y mapas de riesgo.
- Usar modelos más profundos con backbones como ResNet o EfficientNet.

---

Hecho con ❤️ para comunidades que enfrentan desastres naturales.  
