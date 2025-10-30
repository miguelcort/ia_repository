"""
Utilidades para Machine Learning - Clase 1
Diplomado Machine Learning - Universidad Distrital

Este módulo contiene funciones de utilidad para trabajos de ML básicos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def cargar_dataset(ruta, sep=',', encoding='utf-8'):
    """
    Carga un dataset desde un archivo CSV.
    
    Args:
        ruta (str): Ruta al archivo CSV
        sep (str): Separador del archivo (default: ',')
        encoding (str): Codificación del archivo (default: 'utf-8')
    
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        df = pd.read_csv(ruta, sep=sep, encoding=encoding)
        print(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None


def info_basica(df):
    """
    Muestra información básica del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
    """
    print("=" * 50)
    print("INFORMACIÓN BÁSICA DEL DATASET")
    print("=" * 50)
    
    print(f"\nForma: {df.shape}")
    print(f"Número de filas: {df.shape[0]}")
    print(f"Número de columnas: {df.shape[1]}")
    
    print("\n" + "-" * 50)
    print("TIPOS DE DATOS")
    print("-" * 50)
    print(df.dtypes)
    
    print("\n" + "-" * 50)
    print("VALORES FALTANTES")
    print("-" * 50)
    valores_faltantes = df.isnull().sum()
    porcentaje = (valores_faltantes / len(df)) * 100
    resumen = pd.DataFrame({
        'Valores Faltantes': valores_faltantes,
        'Porcentaje': porcentaje
    })
    print(resumen[resumen['Valores Faltantes'] > 0])
    
    print("\n" + "-" * 50)
    print("ESTADÍSTICAS DESCRIPTIVAS")
    print("-" * 50)
    print(df.describe())


def visualizar_distribucion(df, columna, tipo='continua'):
    """
    Visualiza la distribución de una columna.
    
    Args:
        df (pd.DataFrame): Dataset
        columna (str): Nombre de la columna
        tipo (str): 'continua' o 'categorica'
    """
    plt.figure(figsize=(10, 5))
    
    if tipo == 'continua':
        plt.subplot(1, 2, 1)
        plt.hist(df[columna].dropna(), bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Histograma: {columna}')
        plt.xlabel(columna)
        plt.ylabel('Frecuencia')
        
        plt.subplot(1, 2, 2)
        plt.boxplot(df[columna].dropna())
        plt.title(f'Box Plot: {columna}')
        plt.ylabel(columna)
    
    elif tipo == 'categorica':
        conteo = df[columna].value_counts()
        plt.bar(conteo.index, conteo.values)
        plt.title(f'Distribución: {columna}')
        plt.xlabel(columna)
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()


def metricas_clasificacion(y_true, y_pred, nombres_clases=None):
    """
    Calcula métricas de clasificación.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        nombres_clases: Nombres de las clases (opcional)
    
    Returns:
        dict: Diccionario con las métricas
    """
    metricas = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }
    
    print("=" * 50)
    print("MÉTRICAS DE CLASIFICACIÓN")
    print("=" * 50)
    for nombre, valor in metricas.items():
        print(f"{nombre.upper()}: {valor:.4f}")
    
    return metricas


def matriz_correlacion(df, figsize=(12, 8)):
    """
    Visualiza la matriz de correlación del dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        figsize (tuple): Tamaño de la figura
    """
    # Seleccionar solo columnas numéricas
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Calcular correlación
    corr = df_numeric.corr()
    
    # Visualizar
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Matriz de Correlación')
    plt.tight_layout()
    plt.show()


def detectar_outliers(df, columna, metodo='iqr'):
    """
    Detecta outliers en una columna.
    
    Args:
        df (pd.DataFrame): Dataset
        columna (str): Nombre de la columna
        metodo (str): 'iqr' o 'zscore'
    
    Returns:
        pd.Series: Serie booleana indicando outliers
    """
    if metodo == 'iqr':
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
    
    elif metodo == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[columna].dropna()))
        outliers = z_scores > 3
    
    print(f"Outliers detectados en '{columna}': {outliers.sum()}")
    return outliers


if __name__ == "__main__":
    # Ejemplo de uso
    from sklearn.datasets import load_iris
    
    # Cargar dataset de ejemplo
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Usar funciones de utilidad
    info_basica(df)
    visualizar_distribucion(df, 'sepal length (cm)', tipo='continua')
    matriz_correlacion(df)
