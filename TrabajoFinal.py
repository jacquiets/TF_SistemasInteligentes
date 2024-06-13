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

#Carga de datos
df = pd.read_csv('/content/drive/My Drive/dataTest.csv')

# Preprocesamiento
label_encoder = LabelEncoder()
df['SEXO'] = label_encoder.fit_transform(df['SEXO'])
df['C10_NOMBRE'] = label_encoder.fit_transform(df['C10_NOMBRE'])
df['DEPARTAMENTO'] = label_encoder.fit_transform(df['DEPARTAMENTO'])


# Seleccionar características relevantes
X = df[['C10_NOMBRE', 'DEPARTAMENTO', 'EDAD']]


# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Guardar el DataFrame preprocesado en un nuevo archivo (opcional)
preprocessed_df = pd.DataFrame(X_scaled, columns=['C10_NOMBRE', 'DEPARTAMENTO', 'EDAD'])
preprocessed_df.to_csv('/content/drive/My Drive/dataTest_preprocessed.csv', index=False)