import pandas as pd

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