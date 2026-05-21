import streamlit as st
import pandas as pd
import numpy as np
import io
import pickle
from google.cloud import storage
from river import linear_model, preprocessing, metrics

# =========================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# =========================================================
st.set_page_config(page_title="Aprendizaje en línea con River", page_icon="🚕")
st.title("Aprendizaje en línea con River (Corregido)")

st.markdown("""
Este panel demuestra cómo un modelo de **aprendizaje incremental** puede entrenarse y actualizarse 
a partir de un dataset grande alojado en **Google Cloud Storage (GCS)**.  
""")

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def save_model_to_gcs(model, bucket_name, destination_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_string(pickle.dumps(model))
        st.success(f"Modelo guardado en GCS: `{destination_blob}`")
    except Exception as e:
        st.warning(f"No se pudo guardar el modelo: {e}")

def load_model_from_gcs(bucket_name, source_blob):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob)
        if blob.exists():
            data = blob.download_as_bytes()
            st.info("Modelo cargado desde GCS.")
            return pickle.loads(data)
        return None
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo previo: {e}")
        return None

# =========================================================
# CONFIGURACIÓN DE PARÁMETROS
# =========================================================
bucket_name = st.text_input("Nombre del bucket de GCS:", "bucket_131025")
prefix = st.text_input("Carpeta/prefijo dentro del bucket:", "tlc_yellow_trips_2022/")
limite = st.number_input("Número de registros por archivo a procesar:", value=5000, step=100)
mostrar_grafico = st.checkbox("Mostrar gráfico de evolución del R²", value=True)

# Parámetro de estabilidad
RESET_THRESHOLD = -5.0  # Si el R2 cae por debajo de esto, reiniciamos el modelo

# =========================================================
# INICIALIZACIÓN DEL MODELO
# =========================================================
MODEL_PATH = "models/model_incremental.pkl"

if "model" not in st.session_state:
    loaded = load_model_from_gcs(bucket_name, MODEL_PATH)
    if loaded:
        st.session_state.model = loaded
    else:
        # Pipeline: Estandarización -> Regresión Lineal
        st.session_state.model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    
    st.session_state.metric = metrics.R2()
    st.session_state.history = []

model = st.session_state.model
r2 = st.session_state.metric

# =========================================================
# EXTRACCIÓN DE CARACTERÍSTICAS (Simplificada y Robusta)
# =========================================================
def _extract_x(row):
    # Usamos características simples y seguras para evitar explosión de gradientes
    return {
        "dist": float(row["trip_distance"]),
        "pass": float(row["passenger_count"]),
        # Nota: He quitado las transformaciones complejas de tiempo para 
        # garantizar que funcione igual que el segundo código.
    }

# =========================================================
# FUNCIÓN DE STREAMING (CORREGIDA)
# =========================================================
def stream_from_bucket(bucket_name, prefix, limite=5000, chunksize=1000):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    st.info(f"Se encontraron {len(blobs)} archivos en `{prefix}`.")

    # Referenciamos el modelo y métrica globalmente para poder resetearlos
    global model, r2 

    for idx, blob in enumerate(blobs, start=1):
        st.write(f"📦 Procesando archivo {idx}/{len(blobs)}: `{blob.name.split('/')[-1]}`")

        try:
            content = blob.download_as_bytes()
            buffer = io.BytesIO(content)

            count = 0
            
            # Leemos por chunks para eficiencia
            for chunk in pd.read_csv(buffer, chunksize=chunksize, low_memory=False):
                
                # 1. Validación de columnas
                cols_needed = ["trip_distance", "passenger_count", "fare_amount"]
                if not set(cols_needed).issubset(chunk.columns):
                    continue

                # 2. Conversión numérica forzada
                for col in cols_needed:
                    chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

                # 3. FILTRO ESTRICTO (Clave para que no explote el R2)
                # Copiado del código que funciona:
                chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_needed)
                chunk = chunk[
                    (chunk["fare_amount"].between(2, 200)) & 
                    (chunk["trip_distance"].between(0.1, 50)) & 
                    (chunk["passenger_count"].between(1, 6))
                ]

                if chunk.empty:
                    continue

                # 4. Bucle de aprendizaje
                for _, row in chunk.iterrows():
                    if count >= limite:
                        break

                    # Extracción de X e Y
                    x = _extract_x(row)
                    y = float(row["fare_amount"])

                    # Predicción y Aprendizaje
                    y_pred = model.predict_one(x)
                    model.learn_one(x, y)
                    r2.update(y, y_pred)
                    
                    count += 1

                    # 5. MONITORIZACIÓN Y RESET (Salvavidas)
                    # Si el modelo diverge (R2 muy negativo), lo reiniciamos
                    if count > 500 and r2.get() < RESET_THRESHOLD:
                        st.warning(f"⚠️ Modelo inestable detectado (R²={r2.get():.2f}). Reiniciando pesos...")
                        # Reiniciar modelo y métrica
                        model = preprocessing.StandardScaler() | linear_model.LinearRegression()
                        r2 = metrics.R2()
                        st.session_state.model = model
                        st.session_state.metric = r2
                        break # Salimos del chunk actual para empezar limpios

                if count >= limite:
                    break

        except Exception as e:
            st.error(f"Error leyendo `{blob.name}`: {e}")
            continue

        # Devolvemos el nombre y el score actual
        yield blob.name, r2.get()

# =========================================================
# LÓGICA DE ACTUALIZACIÓN
# =========================================================
if st.button("Actualizar modelo con datos del bucket"):
    st.info("Iniciando entrenamiento incremental...")

    progreso = st.progress(0)
    nombres, valores = [], []

    blobs_count = len(list(storage.Client().bucket(bucket_name).list_blobs(prefix=prefix)))
    if blobs_count == 0: blobs_count = 1

    # Iteramos sobre el generador
    for i, (fname, score) in enumerate(stream_from_bucket(bucket_name, prefix, limite)):
        nombres.append(fname.split("/")[-1])
        valores.append(score)
        
        # Actualizar historial en Session State
        st.session_state.history.append(score)
        
        # Barra de progreso
        progreso.progress(min((i + 1) / blobs_count, 1.0))
        st.write(f"R² acumulado tras `{fname}`: **{score:.4f}**")

    progreso.empty()
    st.success("¡Entrenamiento completado!")

    # Guardar modelo actualizado
    save_model_to_gcs(model, bucket_name, MODEL_PATH)

    # Gráfico
    if mostrar_grafico and st.session_state.history:
        st.subheader("Evolución del R²")
        df_hist = pd.DataFrame(st.session_state.history, columns=["R2"])
        st.line_chart(df_hist)

# =========================================================
# ESTADO
# =========================================================
st.markdown("---")
st.write(f"**R² Actual del modelo en memoria:** {r2.get():.4f}")





