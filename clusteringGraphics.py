import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Función para leer un archivo CSV y convertirlo en un DataFrame con columnas específicas
def leer_csv_a_dataframe(ruta_archivo, columnas):
    """
    Lee un archivo CSV y lo convierte en un DataFrame de pandas con columnas específicas.
    
    Parámetros:
    ruta_archivo (str): Ruta al archivo CSV.
    columnas (list): Lista de nombres de columnas a leer.
    
    Retorna:
    DataFrame: Un DataFrame de pandas con los datos del archivo CSV.
    """
    try:
        # Leer el archivo CSV con las columnas especificadas
        df = pd.read_csv(ruta_archivo, usecols=columnas)
        print("Archivo CSV leído exitosamente.")
        return df
    except FileNotFoundError:
        print(f"Error: El archivo en la ruta '{ruta_archivo}' no fue encontrado.")
    except pd.errors.EmptyDataError:
        print("Error: El archivo CSV está vacío.")
    except pd.errors.ParserError:
        print("Error: Hubo un problema al analizar el archivo CSV.")
    except Exception as e:
        print(f"Error inesperado: {e}")

# Ruta al archivo CSV
ruta_archivo = './dataTest.csv'  # Reemplaza con la ruta correcta a tu archivo

# Lista de columnas a leer
columnas = ['C10_NOMBRE', 'DEPARTAMENTO', 'EDAD', 'SEXO']

# Llamar a la función y obtener el DataFrame
df = leer_csv_a_dataframe(ruta_archivo, columnas)

# Mostrar las primeras filas del DataFrame
if df is not None:
    print(df.head())

    # Preprocesamiento de datos
    # Codificación de la variable 'C10_NOMBRE'
    label_encoder = LabelEncoder()
    df['C10_NOMBRE'] = label_encoder.fit_transform(df['C10_NOMBRE'])

    # Normalización de los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[['EDAD', 'C10_NOMBRE']])

    # Determinación del número óptimo de clusters con el método del codo
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)

    # Gráfico del método del codo
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('SSE (Inercia)')
    plt.title('Método del Codo para encontrar el k óptimo')
    plt.show()

    # Seleccionar k (por ejemplo, 3)
    k_optimo = 3
    kmeans = KMeans(n_clusters=k_optimo, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # Visualización de los clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='EDAD', y='C10_NOMBRE', hue='Cluster', palette='viridis')
    plt.title('Clusters de K-means')
    plt.show()

    # Gráfico de distribución de EDAD por cada cluster
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='EDAD', hue='Cluster', multiple='stack', palette='viridis')
    plt.title('Distribución de EDAD por Cluster')
    plt.show()

    # Gráfico de distribución de C10_NOMBRE por cada cluster
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='C10_NOMBRE', hue='Cluster', multiple='stack', palette='viridis')
    plt.title('Distribución de Diagnóstico de Tumor (C10_NOMBRE) por Cluster')
    plt.show()

    # Gráfico de barras de DEPARTAMENTO por cada cluster
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='DEPARTAMENTO', hue='Cluster', palette='viridis')
    plt.title('Número de casos por Departamento y Cluster')
    plt.xticks(rotation=45)
    plt.show()

    # Mostrar los primeros registros del DataFrame con los clusters asignados
    print(df.head())