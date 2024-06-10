import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar google drive
# drive.mount('/content/drive')

# Creación de DataFrame
#data = {
#    'DEPARTAMENTO': ['Lima', 'Lima', 'Arequipa', 'Cusco'],
#    'EDAD': [34, 45, 65, 23],
#    'SEXO': ['F', 'M', 'F', 'M'],
#    'COD_PROCEDIMIENTO': [1001, 1002, 1003, 1004],
#    'VALOR_BRUTO': [150.50, 200.75, 100.00, 300.20],
#}

#df = pd.DataFrame(data)
df = pd.read_csv('./dataTest.csv')

# Preprocesamiento
# Convertir variables categóricas a numéricas
df['SEXO'] = df['SEXO'].map({'FEMENINO': 0, 'MASCULINO': 1})
df['C10_NOMBRE'] = df['C10_NOMBRE'].astype('category')
df['COD_DIAG'] = df['C10_NOMBRE'].cat.codes

# Seleccionar características para el clustering
X = df[['COD_DIAG', 'EDAD','UBIGEO','SEXO']]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-means
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X_scaled)

# Añadir los labels al DataFrame original
df['Cluster'] = kmeans.labels_

# Visualización
plt.figure(figsize=(5, 3))
sns.scatterplot(data=df, x='EDAD', y='C10_NOMBRE', hue='Cluster', palette='viridis')
plt.title('K-means Clustering de Pacientes Oncológicos')
plt.xlabel('Edad')
plt.ylabel('Diagnostico')
plt.show()
