import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client, Client
from cryptography.fernet import Fernet

# Configuración de Supabase y Fernet
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
FERNET_KEY = st.secrets["FERNET_KEY"].encode()

cipher = Fernet(FERNET_KEY)

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Depresión",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🧠 Sistema de Evaluación de Depresión")
st.markdown("""
Esta aplicación combina análisis bayesiano con procesamiento de lenguaje natural para evaluar indicios de depresión 
basándose en información demográfica, antecedentes familiares y análisis de texto.
""")

# Carga de datos
@st.cache_data
def cargar_datos():
    try:
        tabla_antecedentes = pd.read_csv("tablas/tabla_genetico.csv")
        tabla_demografica = pd.read_csv("tablas/tabla_demografica.csv")
        tabla_orientacion_sexual = pd.read_csv("tablas/tabla_social.csv")
        
        # Cargar datos para el modelo de ML
        si = pd.read_csv("tablas/clean_d_tweets.csv")
        no = pd.read_csv("tablas/clean_non_d_tweets.csv")
        
        si2 = si[['id', 'tweet']].copy()
        si2['etiqueta'] = 1
        no2 = no[['id', 'tweet']].copy()
        no2['etiqueta'] = 0
        total = pd.concat([si2, no2], axis=0).dropna()
        
        # Limpieza de texto
        total['tweet'] = total['tweet'].str.lower().str.strip()
        total['tweet'] = total['tweet'].str.replace(r'\s+', ' ', regex=True)
        total['tweet'] = total['tweet'].str.replace(r'http\S+', '', regex=True)
        total['tweet'] = total['tweet'].str.replace(r'@\w+', '', regex=True)
        total['tweet'] = total['tweet'].str.replace(r'#', '', regex=True)
        
        return tabla_antecedentes, tabla_demografica, tabla_orientacion_sexual, total
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None, None, None, None

# Función para procesar CURP
def datos_del_curp(curp: str):
    try:
        if len(curp) != 18:
            raise ValueError("El CURP debe tener 18 caracteres.")
        
        sexo = curp[10].upper()
        año = int(curp[4:6])
        mes = int(curp[6:8])
        dia = int(curp[8:10])
        
        if año <= int(str(datetime.now().year)[-2:]):
            año += 2000
        else:
            año += 1900
            
        fecha = datetime(año, mes, dia)
        edad = (datetime.now() - fecha).days // 365
        
        return sexo, edad, fecha
    except Exception as e:
        st.error(f"Error al procesar CURP: {e}")
        return None, None, None

# Funciones para el cálculo bayesiano
def prevalencia_demografica(sexo, edad, tabla_demografica):
    if edad in tabla_demografica['edad'].values:
        filtro = tabla_demografica[(tabla_demografica['edad'] == edad) & (tabla_demografica['sexo'] == sexo)]
        if not filtro.empty:
            return filtro['prob_base'].values[0]
    return 0.05  # valor por defecto

def orien_sex(orientacion_sexual, tabla_orientacion_sexual):
    if orientacion_sexual == "SI":
        lr = tabla_orientacion_sexual.loc[tabla_orientacion_sexual['orientacion'] == 'heterosexual', 'LR'].values[0]
    else:
        lr = tabla_orientacion_sexual.loc[tabla_orientacion_sexual['orientacion'] == 'homosexual', 'LR'].values[0]
    return lr

def ant_gene(genetico, tabla_antecedentes):
    if genetico == "SI":
        lr = tabla_antecedentes.loc[tabla_antecedentes['antecedente'] == 'si', 'LR'].values[0]
    else:
        lr = tabla_antecedentes.loc[tabla_antecedentes['antecedente'] == 'no', 'LR'].values[0]
    return lr

def odds_prev(p):
    return p / (1 - p)

def odds_post(prev_odds, lr):
    return prev_odds * lr

def prob_post(o):
    return o / (1 + o)

def bayes(p_base, LRs):
    odd = odds_prev(p_base)
    for lr in LRs:
        odd = odds_post(odd, lr)
    return prob_post(odd)

# Entrenamiento del modelo (se ejecuta una vez)
@st.cache_resource
def entrenar_modelo(total):
    X_train, X_test, y_train, y_test = train_test_split(
        total['tweet'], 
        total['etiqueta'], 
        test_size=0.2, 
        random_state=42,
        stratify=total['etiqueta']
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, max_df=0.9)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train_tfidf, y_train)
    
    return vectorizer, clf

# Interfaz principal de la aplicación
def main():
    # Cargar datos
    tabla_antecedentes, tabla_demografica, tabla_orientacion_sexual, total = cargar_datos()
    if tabla_antecedentes is None:
        st.error("No se pudieron cargar los datos necesarios. Verifica las rutas de los archivos.")
        return

    # Sidebar para entrada de datos
    with st.sidebar:
        st.header("📋 Información del Usuario")
        
        # Entrada de CURP
        curp = st.text_input("Ingresa tu CURP:", max_chars=18, help="El CURP debe tener 18 caracteres")
        
        if curp:
            sexo, edad, fecha_nacimiento = datos_del_curp(curp)
            if sexo and edad:
                st.success(f"CURP válido: Sexo: {sexo}, Edad: {edad} años")
                
                # Mostrar información demográfica
                st.subheader("Información Demográfica")
                st.info(f"Sexo: {'Masculino' if sexo == 'H' else 'Femenino'}")
                st.info(f"Edad: {edad} años")
                st.info(f"Fecha de nacimiento: {fecha_nacimiento.strftime('%d/%m/%Y')}")
            else:
                st.error("CURP inválido. Verifica el formato.")
        
        # Espaciador
        st.markdown("---")

    # Contenido principal
    tab1, tab2, tab3 = st.tabs(["📊 Análisis Bayesiano", "📝 Análisis de Texto", "💾 Resultados"])

    with tab1:
        st.header("Análisis Bayesiano de Riesgo")
        
        if not curp or not sexo or not edad:
            st.warning("Ingresa un CURP válido en la barra lateral para continuar.")
        else:
            # Preguntas usando widgets de Streamlit
            st.subheader("Preguntas de Evaluación")
            
            col1, col2 = st.columns(2)
            with col1:
                orientacion_sexual = st.radio(
                    "¿Te consideras heterosexual?",
                    options=["SI", "NO"],
                    index=0,
                    help="Selecciona tu orientación sexual"
                )
            with col2:
                genetico = st.radio(
                    "¿Alguno de tus familiares en primer grado ha tenido depresión?",
                    options=["SI", "NO"],
                    index=1,
                    help="Familiares en primer grado: padres, hermanos o hijos"
                )
            
            # Calcular probabilidad bayesiana
            if st.button("Calcular Probabilidad de Riesgo", type="primary"):
                with st.spinner("Calculando probabilidad..."):
                    p_base = prevalencia_demografica(sexo, edad, tabla_demografica)
                    lr_orientacion = orien_sex(orientacion_sexual, tabla_orientacion_sexual)
                    lr_genetico = ant_gene(genetico, tabla_antecedentes)
                    
                    resultado_bayes = bayes(p_base, [lr_orientacion, lr_genetico])
                    
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis Bayesiano")
                    
                    # Mostrar métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prevalencia Base", f"{p_base:.4f}")
                    with col2:
                        st.metric("Likelihood Ratio Orientación", f"{lr_orientacion:.2f}")
                    with col3:
                        st.metric("Likelihood Ratio Genético", f"{lr_genetico:.2f}")
                    
                    # Mostrar probabilidad final con barra de progreso
                    st.metric("Probabilidad Final de Riesgo", f"{resultado_bayes:.4f}")
                    st.progress(float(resultado_bayes))
                    
                    # Interpretación
                    if resultado_bayes < 0.3:
                        st.success("Riesgo bajo según el análisis bayesiano")
                    elif resultado_bayes < 0.6:
                        st.warning("Riesgo moderado según el análisis bayesiano")
                    else:
                        st.error("Riesgo alto según el análisis bayesiano")
                    
                    # Guardar resultado en session state
                    st.session_state.resultado_bayes = resultado_bayes
                    st.session_state.orientacion_sexual = orientacion_sexual
                    st.session_state.genetico = genetico

    with tab2:
        st.header("Análisis de Texto para Detección de Depresión")
        
        # Entrenar modelo (solo una vez)
        vectorizer, clf = entrenar_modelo(total)
        
        # Área de texto para entrada del usuario
        st.subheader("Describe tu estado de ánimo")
        user_input = st.text_area(
            "Platícame sobre tu estado de ánimo en estas últimas dos semanas:",
            height=150,
            help="Describe tus sentimientos, emociones y estado general"
        )
        
        if st.button("Analizar Texto", type="primary") and user_input:
            with st.spinner("Analizando texto..."):
                # Limpieza del texto
                texto_limpio = user_input.lower()
                texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
                texto_limpio = re.sub(r'http\S+', '', texto_limpio)
                texto_limpio = re.sub(r'@\w+', '', texto_limpio)
                texto_limpio = texto_limpio.replace('#', '')
                
                # Vectorización y predicción
                nuevo_vector = vectorizer.transform([texto_limpio])
                proba = clf.predict_proba(nuevo_vector)[0][1]
                
                # Mostrar resultados
                st.subheader("Resultados del Análisis de Texto")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilidad de Depresión", f"{proba:.4f}")
                with col2:
                    if proba >= 0.5:
                        st.error("El modelo detecta posibles indicios de depresión")
                    else:
                        st.success("El modelo no detecta indicios significativos de depresión")
                
                # Guardar resultado en session state
                st.session_state.resultado_tweet = proba
                st.session_state.texto_analizado = user_input

    with tab3:
        st.header("Resultados Consolidados y Almacenamiento")
        
        if 'resultado_bayes' in st.session_state and 'resultado_tweet' in st.session_state:
            # Mostrar resultados consolidados
            st.subheader("Resumen de Resultados")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidad Bayesiana", f"{st.session_state.resultado_bayes:.4f}")
            with col2:
                st.metric("Probabilidad por Texto", f"{st.session_state.resultado_tweet:.4f}")
            
            # Calcular probabilidad combinada (promedio simple)
            prob_combinada = (st.session_state.resultado_bayes + st.session_state.resultado_tweet) / 2
            st.metric("Probabilidad Combinada", f"{prob_combinada:.4f}")
            
            # Interpretación final
            st.subheader("Interpretación Final")
            if prob_combinada < 0.4:
                st.success("""
                **Bajo riesgo**: Los resultados sugieren un riesgo bajo de depresión. 
                Se recomienda mantener hábitos saludables y realizar chequeos regulares.
                """)
            elif prob_combinada < 0.7:
                st.warning("""
                **Riesgo moderado**: Se detectaron algunos indicios que merecen atención. 
                Considera consultar con un profesional de salud mental para una evaluación más detallada.
                """)
            else:
                st.error("""
                **Alto riesgo**: Los resultados sugieren un riesgo alto de depresión. 
                Se recomienda strongly consultar con un profesional de salud mental para una evaluación completa y apropiada.
                """)
            
            # Botón para guardar resultados
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            
            curp_encriptado = cipher.encrypt(curp.encode()).decode()
            
            # Ejemplo de insertar datos
            def guardar_resultados(curp_encriptado, orientacion, genetico, bayes, tweet, texto):
                data = {
                    "curp": curp_encriptado,
                    "orientacion_sexual": orientacion,
                    "genetico": genetico,
                    "resultado_bayes": bayes,
                    "resultado_tweet": tweet,
                    "texto_analizado": texto,
                }
                supabase.table("resultados_depresion").insert(data).execute()
            
            if st.button("Guardar resultados en la base de datos"):
                guardar_resultados(curp_encriptado, st.session_state.orientacion_sexual, st.session_state.genetico, 
                                 st.session_state.resultado_bayes, st.session_state.resultado_tweet, 
                                 st.session_state.texto_analizado)
                st.success("✅ Resultados guardados en la base de datos")
        else:
            st.info("Complete ambos análisis (Bayesiano y de Texto) para ver los resultados consolidados.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
