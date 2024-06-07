# Identificar patrones segun diagnostico , departamento y edad

# Importar las librerías
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar google drive
from google.colab import drive
drive.mount('/content/drive')

# Creación de DataFrame
data = {
    'DEPARTAMENTO': ['Lima', 'Lima', 'Arequipa', 'Cusco'],
    'EDAD': [34, 45, 65, 23],
    'SEXO': ['F', 'M', 'F', 'M'],
    'COD_PROCEDIMIENTO': [1001, 1002, 1003, 1004],
    'VALOR_BRUTO': [150.50, 200.75, 100.00, 300.20],
}

#df = pd.DataFrame(data)
df = pd.read_csv('/content/drive/My Drive/dataTest.csv')

# Preprocesamiento
# Convertir variables categóricas a numéricas
df['SEXO'] = df['SEXO'].map({'FEMENINO': 0, 'MASCULINO': 1})
df['C10_NOMBRE'] = df['C10_NOMBRE'].astype('category')
df['COD_DIAG'] = df['C10_NOMBRE'].cat.codes
