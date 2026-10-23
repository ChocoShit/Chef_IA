import streamlit as st
import pandas as pd
import numpy as np
import urllib.parse
from deep_translator import GoogleTranslator
from model import NeuralNetwork

# --- 1. CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="Chef IA - NeuroDifuso", page_icon="👩‍🍳", layout="centered")

st.markdown("""
    <style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-image: linear-gradient(rgba(15, 15, 15, 0.85), rgba(15, 15, 15, 0.85)), 
                          url("https://images.unsplash.com/photo-1495521821757-a1efb6729352?q=80&w=2000&auto=format&fit=crop");
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    a.header-anchor, h1 a, h2 a, h3 a { display: none !important; }
    .titulo-app { text-align: center; color: #FF6347; font-weight: 800; font-size: 2.8rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.6); margin-bottom: 0.5rem; }
    .subtitulo { text-align: center; color: #E0E0E0; font-size: 1.1em; margin-bottom: 2rem; }
    div[data-testid="stExpander"] { background-color: rgba(25, 25, 25, 0.7) !important; border-radius: 10px; border: 1px solid #444; }
    
    /* Estilos de las etiquetas de categoría y procedencia de texto limpio */
    .etiqueta-categoria { background-color: #FF6347; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: bold; margin-right: 8px; display: inline-block; margin-bottom: 10px; box-shadow: 1px 1px 3px rgba(0,0,0,0.5);}
    .etiqueta-origen { background-color: #2E7D32; color: white; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: bold; margin-right: 8px; display: inline-block; margin-bottom: 10px; box-shadow: 1px 1px 3px rgba(0,0,0,0.5);}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='titulo-app'>👩‍🍳 Tu Chef Personal con IA</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Selecciona el tipo de platillo, tus ingredientes y la IA te recomendará las mejores opciones.</p>", unsafe_allow_html=True)

# --- 2. CARGA DE DATOS ORIGINALES ---
@st.cache_data
def load_data():
    df = pd.read_csv("epi_r.csv").dropna(subset=['title', 'rating', 'calories', 'protein', 'fat'])
    return df

df = load_data()

# --- 3. DICCIONARIOS ---
diccionario_ingredientes = {
    "Tomate": "tomato", "Pollo": "chicken", "Ajo": "garlic", 
    "Cebolla": "onion", "Carne de res": "beef", "Cerdo": "pork", 
    "Queso": "cheese", "Papa": "potato", "Zanahoria": "carrot", 
    "Leche": "milk", "Huevo": "egg", "Arroz": "rice", 
    "Pasta": "pasta", "Pescado": "fish", "Espinaca": "spinach", 
    "Limón": "lemon", "Mantequilla": "butter", "Albahaca": "basil",
    "Champiñones": "mushroom", "Manzana": "apple"
}

categorias_comida = {
    'breakfast': '🍳 Desayuno', 'lunch': '🥗 Almuerzo', 
    'dinner': '🍽️ Cena', 'dessert': '🍰 Postre', 'snack': '🍿 Merienda'
}

# Diccionario de columnas de origen en el dataset
categorias_origen = {
    'italian': 'Cocina Italiana', 'mexican': 'Cocina Mexicana', 'asian': 'Cocina Asiática', 
    'french': 'Cocina Francesa', 'spanish': 'Cocina Española', 'american': 'Cocina Americana',
    'turkish': 'Cocina Turca', 'indian': 'Cocina India', 'japanese': 'Cocina Japonesa', 
    'chinese': 'Cocina China', 'british': 'Cocina Británica', 'greek': 'Cocina Griega',
    'peruvian': 'Cocina Peruana', 'moroccan': 'Cocina Marroquí', 'german': 'Cocina Alemana',
    'irish': 'Cocina Irlandesa', 'jewish': 'Cocina Judía', 'middle eastern': 'Cocina de Medio Oriente'
}

def obtener_etiquetas_html(row):
    html_cat = ""
    html_ori = ""
    
    # Buscar categoría de comida
    for col, nombre in categorias_comida.items():
        if col in row.index and row[col] == 1.0:
            html_cat += f"<span class='etiqueta-categoria'>{nombre}</span>"
            
    # Buscar procedencia real en las columnas del dataset
    for col, nombre in categorias_origen.items():
        if col in row.index and row[col] == 1.0:
            html_ori += f"<span class='etiqueta-origen'>📍 {nombre}</span>"
            break # Tomar la primera procedencia detectada
            
    # Valores por defecto si la base de datos no tiene esa información específica
    if html_cat == "":
        html_cat = "<span class='etiqueta-categoria'>🍲 Platillo General</span>"
    if html_ori == "":
        html_ori = "<span class='etiqueta-origen'>📍 Cocina Internacional</span>"
        
    return html_cat + html_ori

# --- 4. FUNCIONES DE LÓGICA DIFUSA ---
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a + 1e-5), (c - x) / (c - b + 1e-5)))

# --- 5. INTERFAZ Y FILTROS SIMPLIFICADOS ---
st.subheader("🍽️ Paso 1: ¿Qué tipo de platillo buscas?")
tipo_comida_seleccionada = st.selectbox(
    "Selecciona la categoría:",
    ["Cualquiera", "Desayuno", "Almuerzo", "Cena", "Postre", "Merienda"]
)

st.subheader("🛒 Paso 2: ¿Qué ingredientes tienes en casa?")
ingredientes_seleccionados = st.multiselect(
    "Elige uno o varios ingredientes:",
    options=list(diccionario_ingredientes.keys()),
    placeholder="Ej. Pollo, Tomate, Ajo..."
)

cantidad_recetas = st.slider("¿Cuántas opciones quieres ver?", min_value=1, max_value=10, value=3)

if st.button("✨ Generar Menú Inteligente", use_container_width=True):
    if ingredientes_seleccionados:
        recetas_filtradas = df.copy()
        
        # --- FILTRO 1: CATEGORÍA ---
        mapa_columnas_cat = {"Desayuno": "breakfast", "Almuerzo": "lunch", "Cena": "dinner", "Postre": "dessert", "Merienda": "snack"}
        if tipo_comida_seleccionada != "Cualquiera":
            columna_cat = mapa_columnas_cat[tipo_comida_seleccionada]
            if columna_cat in recetas_filtradas.columns:
                recetas_filtradas = recetas_filtradas[recetas_filtradas[columna_cat] == 1.0]
        
        # --- FILTRO 2: INGREDIENTES ---
        columnas_ingles = [diccionario_ingredientes[ing] for ing in ingredientes_seleccionados]
        for col in columnas_ingles:
            if col in recetas_filtradas.columns:
                recetas_filtradas = recetas_filtradas[recetas_filtradas[col] == 1.0]
            else:
                recetas_filtradas = recetas_filtradas.head(0)
        
        if len(recetas_filtradas) > 0:
            st.success(f"¡Genial! Encontramos {len(recetas_filtradas)} recetas que encajan. Evaluando con IA...")
            
            # --- FUZZIFICACIÓN ---
            recetas_filtradas['cal_baja'] = trimf(recetas_filtradas['calories'], 0, 0, 400)
            recetas_filtradas['cal_media'] = trimf(recetas_filtradas['calories'], 250, 500, 800)
            recetas_filtradas['cal_alta'] = trimf(recetas_filtradas['calories'], 600, 1200, 2000)
            recetas_filtradas['prot_baja'] = trimf(recetas_filtradas['protein'], 0, 0, 15)
            recetas_filtradas['prot_media'] = trimf(recetas_filtradas['protein'], 10, 25, 45)
            recetas_filtradas['prot_alta'] = trimf(recetas_filtradas['protein'], 35, 60, 100)
            recetas_filtradas['grasa_baja'] = trimf(recetas_filtradas['fat'], 0, 0, 15)
            recetas_filtradas['grasa_media'] = trimf(recetas_filtradas['fat'], 10, 25, 45)
            recetas_filtradas['grasa_alta'] = trimf(recetas_filtradas['fat'], 35, 60, 100)
            
            # --- PREDICCIÓN CON RED NEURONAL ---
            columnas_entrada = ['cal_baja', 'cal_media', 'cal_alta', 'prot_baja', 'prot_media', 'prot_alta', 'grasa_baja', 'grasa_media', 'grasa_alta']
            input_batch = recetas_filtradas[columnas_entrada].values
            
            nn = NeuralNetwork()
            recetas_filtradas['Rating_IA'] = nn.calculate_batch_output(input_batch)
            top_recetas = recetas_filtradas.sort_values(by='Rating_IA', ascending=False).head(cantidad_recetas)
            
            st.markdown("---")
            st.subheader("🏆 Tus Mejores Opciones")
            
            traductor = GoogleTranslator(source='en', target='es')
            
            for idx, row in top_recetas.iterrows():
                try:
                    titulo_espanol = traductor.translate(row['title'])
                except Exception:
                    titulo_espanol = row['title']
                
                query = urllib.parse.quote(row['title'] + " recipe")
                link_oficial = f"https://www.google.com/search?q={query}"
                
                # Generar etiquetas limpias de categoría y origen basada en el dataset
                html_etiquetas = obtener_etiquetas_html(row)
                
                with st.container():
                    st.markdown(f"### 🍽️ {titulo_espanol}")
                    st.markdown(html_etiquetas, unsafe_allow_html=True)
                    st.markdown(f"**Recomendación de la IA:** ⭐ {row['Rating_IA']:.2f} / 5.0")
                    st.markdown(f"[👉 **Haz clic aquí para ver los pasos de preparación**]({link_oficial})")
                    
                    with st.expander("📊 Ver detalles nutricionales"):
                        st.write(f"🔥 **Calorías:** {row['calories']} kcal")
                        st.write(f"🥩 **Proteínas:** {row['protein']} g")
                        st.write(f"🧈 **Grasas:** {row['fat']} g")
                        st.write(f"*(Título original: {row['title']})*")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
        else:
            st.warning("Lo sentimos, no encontramos platillos que encajen con esos ingredientes para esa categoría. ¡Prueba seleccionando otra categoría o quitando un ingrediente!")
    else:
        st.info("👆 Por favor, selecciona al menos un ingrediente para empezar.")