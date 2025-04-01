# Predicción del Precio de Ventas de Casas en el Condado de King, Washington

## Descripción del Problema
Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje supervisado para predecir el precio de venta de casas en el condado de King, Washington, desde mayo de 2014 hasta mayo desde 2015. La predicción se basará en distintas variables y características de las viviendas contenidas en el dataset.

## Dataset
El dataset utilizado en este proyecto ha sido extraído de la plataforma Kaggle. Contiene información sobre las propiedades vendidas en el condado de King, Washington, durante el periodo mencionado. 

- **Fuente:** [King County House Sales Dataset - Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)
- **Público:** Sí
- **Descripción:** El dataset incluye variables como el precio de venta, número de habitaciones, baños, superficie construida, superficie del terreno, año de construcción, ubicación geográfica, entre otros.

## Solución Adoptada
Para abordar este problema, se seguirá el siguiente enfoque:
1. **Exploración de Datos (EDA):** Se analizarán las distribuciones de las variables, se identificarán valores atípicos y se realizará un preprocesamiento adecuado de los datos.
2. **Preprocesamiento:** Se manejarán valores nulos, se transformarán variables categóricas y se escalarán las variables numéricas si es necesario.
3. **Entrenamiento del Modelo:** Se probarán distintos modelos de regresión, como regresión lineal, regresión con regularización y modelos basados en árboles (Random Forest, XGBoost) para evaluar su desempeño.
4. **Evaluación y Ajuste:** Se medirá la precisión del modelo utilizando métricas como el RMSE y el R^2, ajustando hiperparámetros para optimizar los resultados.
5. **Conclusiones y Recomendaciones:** Se interpretarán los resultados y se evaluará la aplicabilidad del modelo para predicciones futuras.

## Estructura del Repositorio
```
/
├── data/                  # Contiene el dataset original y versiones procesadas
├── img/                   # Imágenes de apoyo y visualizaciones generadas durante el análisis
├── models/                # Modelos entrenados y archivos de configuración
├── notebooks/             # Jupyter Notebooks con la limpieza, análisis y desarrollo del modelo
├── results_notebook/      # Notebook con todo el proceso y resultados de las predicciones y evaluaciones
├── utils/                 # Funciones auxiliares y herramientas útiles
├── README.md              # Documento explicativo del proyecto
```

Este repositorio proporcionará todos los archivos necesarios para la replicación y mejora del modelo de predicción.