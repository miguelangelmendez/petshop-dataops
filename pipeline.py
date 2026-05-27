import pandas as pd
import logging

# configuracion del log (evidencia de monitoreo)
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("iniciando el pipeline dataops...")

try:
    # 1. ingesta
    df = pd.read_csv('ventas_raw.csv')
    logging.info("ingesta exitosa. filas cargadas: " + str(len(df)))

    # 2. limpieza
    df['producto'] = df['producto'].str.lower() # todo a minusculas
    logging.info("limpieza completada: textos estandarizados.")

    # 3. validacion (reglas de negocio)
    # filtramos para dejar solo cantidades mayores a 0 y precios mayores a 0
    filas_validas = (df['cantidad'] > 0) & (df['precio'] > 0)
    
    # anotamos en el log si encontramos errores
    errores = df[~filas_validas]
    for idx, row in errores.iterrows():
        logging.warning(f"anomalia detectada en venta id {row['id_venta']}: datos invalidos.")

    df_limpio = df[filas_validas]

    # 4. carga
    df_limpio.to_csv('ventas_procesadas.csv', index=False)
    logging.info("carga completada con exito. archivo ventas_procesadas.csv creado.")
    print("pipeline ejecutado con exito. revisa app.log y ventas_procesadas.csv")

except Exception as e:
    logging.error(f"error critico en el pipeline: {str(e)}")