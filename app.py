'''
Chef IA - Planificador Inteligente de Recetas
Version 2.0 - Simplificado y con integración del modelo 9-10-1
'''

import streamlit as st
import pandas as pd
import numpy as np
from model_9_10_1 import NeuralNetwork_9_10_1
import os

# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Chef IA — Planificador Inteligente",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS para eliminar barra superior y mejorar estética
st.markdown("""
<style>
    /* Eliminar barra de menú principal */
    .stMenu {visibility: hidden;}
    /* Footer oculto */
    footer {visibility: hidden;}

    .titulo-principal {
        text-align: center;
        color: #FF6347;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .subtitulo {
        text-align: center;
        color: #E0E0E0;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    .recipe-card {
        background: rgba(30, 30, 30, 0.85);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .recipe-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
    }

    .badge-country {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
        color: white;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-rating {
        background: linear-gradient(135deg, #FF6347, #FF4528);
        color: white;
        padding: 5px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-meal {
        background: linear-gradient(135deg, #2196F3, #1976D2);
        color: white;
        padding: 4px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CARGA DE DATOS (optimizada)
# =============================================================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("recipes_internacional.csv")
    except FileNotFoundError:
        return None

    # Limpieza segura
    df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(0)
    df['protein'] = pd.to_numeric(df['protein'], errors='coerce').fillna(0)
    df['fat'] = pd.to_numeric(df['fat'], errors='coerce').fillna(0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)

    if 'cuisine' not in df.columns:
        df['cuisine'] = 'Internacional'
    df['cuisine'] = df['cuisine'].fillna('Internacional')

    return df

df = load_data()

if df is None:
    st.error("⚠️ No se pudieron cargar los datos de recetas.")
    st.stop()

# =============================================================================
# INICIALIZAR MODELO (solo una vez)
# =============================================================================
if 'rating_model' not in st.session_state:
    try:
        st.session_state.rating_model = NeuralNetwork_9_10_1()
    except Exception as e:
        st.warning(f"⚠️ No se pudo cargar el modelo IA: {str(e)[:50]}")
        st.session_state.rating_model = None

model = st.session_state.rating_model

# =============================================================================
# INTERFAZ PRINCIPAL - 3 secciones simplificadas
# =============================================================================

st.markdown("<h1 class='titulo-principal'>🍳 Chef IA</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Tu asistente inteligente para recetas personalizadas</p>", unsafe_allow_html=True)

# Sección 1: Búsqueda rápida
st.markdown("---")
st.subheader("🔍 Búsqueda rápida")

col1, col2 = st.columns(2)
with col1:
    nombre_receta = st.text_input("Nombre de la receta", placeholder="Ej: pasta, tortilla...")
with col2:
    busqueda_pais = st.selectbox(
        "País de origen",
        ["Todos", "Italiana", "Mexicana", "Española", "Americana",
         "Francesa", "Alemana", "China", "Japonesa", "India", "Internacional"]
    )

# Sección 2: Filtros claros
st.markdown("---")
st.subheader("📌 Filtrar por tipo de comida")

 tipo_comida = st.multiselect(
        "¿Para qué momento te sirve?",
        ["Desayuno", "Almuerzo", "Cena", "Merienda", "Snack"],
        default=["Desayuno", "Almuerzo", "Cena"]
    )

# Sección 3: Botón principal
st.markdown("---")

if st.button("✨ Generar recomendaciones", type="primary", use_container_width=True):
    if not nombre_receta and busqueda_pais == "Todos":
        st.warning("📝 Por favor, ingresa un nombre de receta o selecciona un país")
    else:
        # Filtrar datos
        df_filtrado = df.copy()

        if nombre_receta:
            df_filtrado = df_filtrado[df_filtrado['name'].str.contains(nombre_receta, case=False, na=False)]

        if busqueda_pais != "Todos":
            df_filtrado = df_filtrado[df_filtrado['cuisine'].str.lower() == busqueda_pais.lower()]

        # Aplicar filtro de tipo de comida
        df_filtrado = df_filtrado[df_filtrado['meal_type'].isin(tipo_comida) | (df_filtrado['meal_type'].isna())]

        if len(df_filtrado) == 0:
            st.info("No encontré recetas con esos filtros. Intenta con otros términos.")
        else:
            # Mostrar resultados
            st.markdown("---")
            st.subheader(f"🍽️ Recetas encontradas ({len(df_filtrado)})")

            for idx, (_, row) in enumerate(df_filtrado.head(5).iterrows()):
                # Calcular variables fuzzy
                fuzzy_inputs = calcula_fuzzy(row)

                # Calcular rating IA si hay modelo
                if model:
                    try:
                        rating_ia = model.predict(fuzzy_inputs)
                    except:
                        rating_ia = row.get('rating', 4.0)
                else:
                    rating_ia = row.get('rating', 4.0)

                rating_real = round(float(row.get('rating', 4.0)), 1)
                cal = int(float(row.get('calories', 0)))
                prot = int(float(row.get('protein', 0)))
                gras = int(float(row.get('fat', 0)))
                pais = row.get('cuisine', 'Internacional')
                tiempo = row.get('cook_time', 'N/A')

                st.markdown(f"""
                <div class='recipe-card'>
                    <span class='badge-country'>📍 {pais}</span>
                    <span class='badge-rating'>⭐ {rating_ia:.1f} (IA)</span>
                    <h3 style='margin-top: 10px; color: #FFFFFF;'>{row.get('name', 'Receta')} {idx+1}</h3>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Calorías", f"{cal} kcal")
                c2.metric("Proteína", f"{prot} g")
                c3.metric("Grasas", f"{gras} g")

# =============================================================================
# FUNCIÓN DE MEMBRESÍA TRIANGULAR
# =============================================================================
def calcula_fuzzy(row):
    """Calcula las 9 variables fuzzy para una receta"""
    cal = float(row.get('calories', 200))
    prot = float(row.get('protein', 15))
    gras = float(row.get('fat', 10))

    # Triángulos de membresía
    cal_baja = 1.0 if cal < 120 else (0.5 if cal < 250 else 0.0)
    cal_media = 0.6 if 120 <= cal <= 350 else (0.3 if cal < 500 else 0.0)
    cal_alta = 1.0 if cal > 350 else (0.5 if cal > 250 else 0.0)

    prot_baja = 1.0 if prot < 8 else (0.6 if prot < 18 else 0.0)
    prot_media = 0.6 if 8 <= prot <= 25 else (0.3 if prot < 35 else 0.0)
    prot_alta = 1.0 if prot > 25 else (0.5 if prot > 15 else 0.0)

    gras_baja = 1.0 if gras < 4 else (0.6 if gras < 12 else 0.0)
    grasa_media = 0.6 if 4 <= gras <= 15 else (0.3 if gras < 22 else 0.0)
    grasa_alta = 1.0 if gras > 15 else (0.5 if gras > 8 else 0.0)

    return [cal_baja, cal_media, cal_alta,
            prot_baja, prot_media, prot_alta,
            gras_baja, grasa_media, grasa_alta]


# =============================================================================
# BACKWARD COMPATIBILITY - Mantener funciones antiguas (para referencia)
# =============================================================================
def load_data():
    """Alias para compatibilidad hacia atrás"""
    return st.session_state.get('df_loaded', pd.DataFrame())