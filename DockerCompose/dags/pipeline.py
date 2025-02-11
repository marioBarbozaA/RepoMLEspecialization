from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 3, 26),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cars_price_predictionsss',
    default_args=default_args,
    description='Pipeline con regresión lineal para predecir precio de automóviles',
    schedule_interval=timedelta(days=1),  # Corre una vez al día
)

def load_data(**context):
    """
    1. Lee el archivo Excel 'FabricaAutomoviles.xlsx'
    2. Imprime la forma del DataFrame
    3. Devuelve el DataFrame como dict vía XCom
    """
    print("=== [LOAD_DATA] Iniciando la carga de datos ===")
    
    file_path = '/opt/airflow/dags/data/FabricaAutomoviles.xlsx'
    df = pd.read_excel(file_path)
    

    print(f"[LOAD_DATA] Se han cargado {df.shape[0]} filas y {df.shape[1]} columnas.")
    print("[LOAD_DATA] Columnas encontradas:", df.columns.tolist())

    # Enviamos el DataFrame como un dict a XCom
    context['ti'].xcom_push(key='df', value=df.to_dict(orient='list'))
    print("=== [LOAD_DATA] Carga de datos finalizada. ===")


def preprocess_data(**context):
    """
    1. Toma el DataFrame de XCom
    2. Elimina filas nulas
    3. Selecciona todas las columnas como features (excepto la variable target)
    4. Convierte la columna 'tracción' a dummies
    5. Realiza train_test_split
    6. Devuelve los splits a XCom
    """
    print("=== [PREPROCESS_DATA] Iniciando preprocesamiento ===")
    df_dict = context['ti'].xcom_pull(task_ids='load_data', key='df')
    df = pd.DataFrame(df_dict)
    
    # Eliminamos nulos
    df.dropna(inplace=True)
    print(f"[PREPROCESS_DATA] Después de dropna: {df.shape[0]} filas.")

    # DEFINIMOS FEATURES Y TARGET:
    # Todas las columnas relevantes, excepto 'precio_promedio' (nuestra Y)
    features = [
        'millas_por_galon_carretera',
        'cilindros',
        'litros_motor',
        'caballos_fuerza',
        'revoluciones_por_minuto',
        'capacidad_tanque',
        'capacidad_pasajeros',
        'longitud',
        'ancho',
        'peso_en_libras',
        'numero_de_airbags',
        'tracción',            # <--- Columna categórica: delantera, trasera, doble tracción
        'transmisión_manual'
    ]
    target = 'precio_promedio'

    # Separamos X e y
    X = df[features].copy()
    y = df[target].copy()

    # Convertimos la columna 'tracción' en dummies (tendrá 2 columnas si hay 3 categorías)
    X = pd.get_dummies(X, columns=['tracción'], drop_first=True)

    # Si 'transmisión_manual' no es 0/1, sino strings, conviene asegurarnos:
    # X['transmisión_manual'] = X['transmisión_manual'].astype(int)

    print("[PREPROCESS_DATA] Columnas de X luego de get_dummies:")
    print(X.columns.tolist())

    # Dividimos datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[PREPROCESS_DATA] X_train: {X_train.shape}, y_train: {len(y_train)}")
    print(f"[PREPROCESS_DATA] X_test:  {X_test.shape}, y_test:  {len(y_test)}")

    # Guardamos los splits en un dict, para luego push a XCom
    data_splits = {
        'X_train': X_train.to_dict(orient='list'),
        'y_train': y_train.tolist(),
        'X_test': X_test.to_dict(orient='list'),
        'y_test': y_test.tolist()
    }
    context['ti'].xcom_push(key='data_splits', value=data_splits)
    print("=== [PREPROCESS_DATA] Preprocesamiento finalizado. ===")


def train_model(**context):
    """
    1. Recupera splits de XCom
    2. Entrena la regresión lineal múltiple
    3. Imprime coeficientes
    4. Guarda el modelo
    """
    print("=== [TRAIN_MODEL] Iniciando entrenamiento ===")
    data_splits = context['ti'].xcom_pull(task_ids='preprocess_data', key='data_splits')
    
    X_train = pd.DataFrame(data_splits['X_train'])
    y_train = data_splits['y_train']

    # Creamos y entrenamos el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Mostramos coeficientes en log
    print("[TRAIN_MODEL] Coeficientes del modelo:")
    for feature_name, coef in zip(X_train.columns, model.coef_):
        print(f"   {feature_name}: {coef:.4f}")
    print(f"[TRAIN_MODEL] Intercepto: {model.intercept_:.4f}")

    # Guardamos el modelo en un archivo joblib
    model_path = '/opt/airflow/dags/data/modelo_regresion_lineal.joblib'
    joblib.dump(model, model_path)
    print(f"[TRAIN_MODEL] Modelo guardado en: {model_path}")

    # Almacenamos la ruta en XCom por si se necesita en otra tarea
    context['ti'].xcom_push(key='model_path', value=model_path)
    print("=== [TRAIN_MODEL] Entrenamiento finalizado. ===")


def evaluate_model(**context):
    """
    1. Carga el modelo desde la ruta en XCom
    2. Predice usando X_test
    3. Imprime RMSE y R2
    """
    print("=== [EVALUATE_MODEL] Iniciando evaluación ===")
    data_splits = context['ti'].xcom_pull(task_ids='preprocess_data', key='data_splits')
    X_test = pd.DataFrame(data_splits['X_test'])
    y_test = data_splits['y_test']

    model_path = context['ti'].xcom_pull(task_ids='train_model', key='model_path')
    model = joblib.load(model_path)

    # Predicción
    y_pred = model.predict(X_test)

    # Métricas
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"[EVALUATE_MODEL] RMSE: {rmse:.2f}")
    print(f"[EVALUATE_MODEL] R2: {r2:.2f}")
    print("=== [EVALUATE_MODEL] Evaluación finalizada. ===")


# Definimos las tareas de Airflow con PythonOperator
load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag
)

# Orden de ejecución de tareas
load_data_task >> preprocess_data_task >> train_model_task >> evaluate_model_task
