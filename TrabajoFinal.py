# Identificar patrones segun diagnostico , departamento y edad

# Importar las librerías
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

def metodo_codo(data):
  # Determinación del número óptimo de clusters con el método del codo
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # Gráfico del método del codo
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('SSE (Inercia)')
    plt.title('Método del Codo para encontrar el k óptimo')
    plt.show()

def leer_csv_a_dataframe(ruta_archivo, nombre_archivo, columnas):
    """
    Lee un archivo CSV y lo convierte en un DataFrame de pandas con columnas específicas.

    Parámetros:
    ruta_archivo (str): Ruta al archivo en drive.
    nombre_archivo (str): Nombre del archivo CSV
    columnas (list): Lista de nombres de columnas a leer.

    Retorna:
    DataFrame: Un DataFrame de pandas con los datos del archivo CSV.
    """
    try:
        link = ruta_archivo
        # Recuperar archivo de google drive
        !gdown --id {link.split("/")[-2]}
        # Leer el archivo CSV con las columnas especificadas
        df = pd.read_csv(nombre_archivo, usecols=columnas, encoding='latin-1')
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


link_drive = 'https://drive.google.com/file/d/1V1fo2b_DVUhfhU5HeTwV_Wg94HGR6hEt/view?usp=sharing'
nombre_archivo = 'dataSIS.csv'
columnas = ['C10_NOMBRE', 'DEPARTAMENTO', 'EDAD', 'SEXO']

df = leer_csv_a_dataframe(link_drive,nombre_archivo,columnas)

#Si el dataframe no es nulo
if df is not None:
  # Preprocesamiento
  # Convertir variables categóricas a numéricas
  label_encoder = LabelEncoder()
  df['SEXO'] = label_encoder.fit_transform(df['SEXO'])
  df['C10_NOMBRE'] = label_encoder.fit_transform(df['C10_NOMBRE'])
  df['DEPARTAMENTO'] = label_encoder.fit_transform(df['DEPARTAMENTO'])



  # Seleccionar características para el clustering
  X = df[['C10_NOMBRE', 'EDAD','DEPARTAMENTO','SEXO']]

  metodo_codo(df)

  # Escalar los datos
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Aplicar K-means
  kmeans = KMeans(n_clusters=3, random_state=50)
  kmeans.fit(X_scaled)

  # Añadir los labels al DataFrame original
  df['Cluster'] = kmeans.labels_

  # Visualización
  plt.figure(figsize=(10, 6))
  sns.scatterplot(data=df, x='EDAD', y='DEPARTAMENTO', hue='Cluster', palette='viridis')
  plt.title('K-means Clustering de Pacientes Oncológicos')
  plt.xlabel('Edad')
  plt.ylabel('Diagnostico')
  plt.show()

  # print(df)
