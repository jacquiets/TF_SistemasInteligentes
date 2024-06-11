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

df = pd.read_csv('/content/drive/My Drive/dataTest.csv')

# Preprocesamiento
label_encoder = LabelEncoder()
df['SEXO'] = label_encoder.fit_transform(df['SEXO'])
df['C10_NOMBRE'] = label_encoder.fit_transform(df['C10_NOMBRE'])
df['DEPARTAMENTO'] = label_encoder.fit_transform(df['DEPARTAMENTO'])

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
