import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Petshop DataOps", layout="wide")

st.title("🐾 Petshop DataOps - Panel de Monitoreo")
st.subheader("Visualización del Pipeline de Datos en Tiempo Real")

col1, col2 = st.columns(2)

with col1:
    st.header("📥 Datos Crudos (ventas_raw.csv)")
    if os.path.exists("ventas_raw.csv"):
        df_raw = pd.read_csv("ventas_raw.csv")
        st.dataframe(df_raw, use_container_width=True)
    else:
        st.error("No se encontró el archivo ventas_raw.csv")

with col2:
    st.header("🧼 Datos Procesados (ventas_procesadas.csv)")
    if os.path.exists("ventas_procesadas.csv"):
        df_proc = pd.read_csv("ventas_procesadas.csv")
        st.dataframe(df_proc, use_container_width=True)
        st.success("¡Datos limpios y estandarizados !")
    else:
        st.info("El pipeline aún no ha guardado datos limpios.")

st.divider()
st.header("📋 Historial de Auditoría y Alertas (app.log)")
if os.path.exists("app.log"):
    with open("app.log", "r") as f:
        log_lines = f.readlines()
    
    for line in log_lines:
        if "WARNING" in line:
            st.error(line.strip())
        else:
            st.info(line.strip())
