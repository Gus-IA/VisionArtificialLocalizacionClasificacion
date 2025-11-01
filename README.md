# Detección de Objetos con Pytorch y VOC Dataset

Este proyecto demuestra cómo cargar el conjunto de datos **PASCAL VOC**, visualizar las anotaciones de los objetos (bounding boxes) y construir una red neuronal convolucional (CNN) simple para predecir **la localización y clase** de objetos en imágenes.

---

## 🧠 Contenido Aprendido

### 1. **Carga y exploración de datos**
Se utiliza `torchvision.datasets.VOCDetection` para descargar y cargar el dataset PASCAL VOC.  
Se implementan funciones auxiliares para:
- Extraer imágenes y anotaciones.
- Convertir bounding boxes entre formato absoluto y normalizado.
- Visualizar las anotaciones en las imágenes usando `matplotlib`.

### 2. **Visualización de anotaciones**
La función `plot_anns` muestra los *bounding boxes* y etiquetas sobre las imágenes, permitiendo verificar visualmente los datos.

### 3. **Normalización y desnormalización**
Se implementan funciones:
- `norm(bb, shape)`: convierte las coordenadas a valores normalizados entre 0 y 1.
- `unnorm(bb, shape)`: restaura los valores normalizados a coordenadas originales.

### 4. **Aumentación de datos**
Se utiliza la librería **Albumentations** para aplicar transformaciones como `Resize`, manteniendo las etiquetas de los objetos coherentes con los nuevos tamaños.

### 5. **Construcción del modelo**
Se define una clase `Model` basada en PyTorch:
- Bloques convolucionales (`block`) para extraer características.
- Capas lineales (`block2`) para procesar las características y producir dos salidas:
  - `x_loc`: coordenadas normalizadas del bounding box.
  - `x_cls`: clase del objeto detectado.

### 6. **Entrenamiento**
La función `fit` entrena el modelo con una imagen de ejemplo:
- Se usa **L1 Loss** para la regresión de las coordenadas.
- Se usa **CrossEntropyLoss** para la clasificación del objeto.
- El optimizador empleado es **Adam**.

### 7. **Evaluación y visualización de predicciones**
Después del entrenamiento, el modelo predice el bounding box y clase, los cuales se visualizan sobre la imagen.

---
🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.

