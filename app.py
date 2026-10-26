import streamlit as st
import pandas as pd
import numpy as np

# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
st.set_page_config(page_title="Chef IA — Planificador Inteligente", page_icon="🍳", layout="centered")

st.markdown("""
<style>
    /* Estilo del fondo general */
    .stApp {
        background-image: linear-gradient(rgba(15, 15, 15, 0.88), rgba(15, 15, 15, 0.88)),
            url("https://images.unsplash.com/photo-1495521821757-a1efb6729352?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Encabezados y títulos */
    .titulo-principal { text-align: center; color: #FF6347; font-weight: 800; font-size: 2.5rem; margin-bottom: 0.2rem; }
    .subtitulo { text-align: center; color: #E0E0E0; font-size: 1.05rem; margin-bottom: 1.5rem; }
    
    /* Tarjetas de recetas resultado */
    .recipe-card {
        background: rgba(30, 30, 30, 0.75);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    /* Badges / Etiquetas */
    .badge-country {
        background-color: #2E7D32;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-rating {
        background-color: #FF6347;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 2. CARGA SEGURA DE DATOS
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("epi_r_combinado.csv")
    except FileNotFoundError:
        df = pd.read_csv("epi_r.csv")
    
    # Limpieza previa de valores numéricos para evitar ValueError
    df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(0)
    df['protein'] = pd.to_numeric(df['protein'], errors='coerce').fillna(0)
    df['fat'] = pd.to_numeric(df['fat'], errors='coerce').fillna(0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
    
    if 'cuisine' not in df.columns:
        df['cuisine'] = 'Internacional'
    else:
        df['cuisine'] = df['cuisine'].fillna('Internacional')
        
    return df

df = load_data()

# Control de pasos (Wizard)
if "step" not in st.session_state:
    st.session_state.step = 1

st.markdown("<h1 class='titulo-principal'>🍳 Chef IA — Planificador Inteligente</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Diseña tu menú semanal personalizado en pocos pasos</p>", unsafe_allow_html=True)

# Barra de progreso
TOTAL_PASOS = 8
progress_val = int((st.session_state.step / TOTAL_PASOS) * 100)
st.progress(progress_val)
st.caption(f"**Paso {st.session_state.step} de {TOTAL_PASOS}**")

st.markdown("---")

# ---------------------------------------------------------
# PASO 1: PERSONAS Y PRESUPUESTO
# ---------------------------------------------------------
if st.session_state.step == 1:
    st.subheader("👥 1. ¿Para cuántas personas cocinas?")
    personas = st.number_input("Número de comensales:", min_value=1, max_value=12, value=2, step=1)
    st.session_state["personas"] = personas

    st.subheader("💶 Presupuesto semanal estimado")
    presupuesto = st.slider("Selecciona tu rango aproximado (€):", 20, 200, 60, step=5)
    st.session_state["presupuesto"] = presupuesto

    if st.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 2
        st.rerun()

# ---------------------------------------------------------
# PASO 2: ORIGEN Y COCINA DEL MUNDO
# ---------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("🌍 2. ¿Qué origen de cocina prefieres?")
    
    paises_unicos = sorted([c for c in df["cuisine"].unique() if c != "Internacional"])
    opciones_pais = ["Todos los países"] + paises_unicos
    
    pais_sel = st.selectbox("Selecciona un país u origen gastronómico:", opciones_pais)
    st.session_state["pais"] = pais_sel

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 3
        st.rerun()

# ---------------------------------------------------------
# PASO 3: OBJETIVO Y ESTILO DE COMIDA
# ---------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("🎯 3. ¿Qué estilo de comida buscas?")
    
    estilo = st.multiselect(
        "Elige tus preferencias de menú (puedes marcar varias):",
        ["Rápido y fácil", "Bajo en calorías", "Favoritos en familia", "Plato fuerte / Cena", "Snack / Desayuno"],
        default=["Rápido y fácil"]
    )
    st.session_state["estilo"] = estilo

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 2
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 4
        st.rerun()

# ---------------------------------------------------------
# PASO 4: PREFERENCIAS PROTEICAS
# ---------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("🥩 4. ¿Qué fuentes de proteína prefieres?")
    
    proteinas = st.multiselect(
        "Selecciona tus fuentes de proteína habituales:",
        ["pollo", "carne", "cerdo", "pescado", "huevo", "queso"],
        default=["pollo", "carne"]
    )
    st.session_state["proteinas"] = proteinas

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 3
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 5
        st.rerun()

# ---------------------------------------------------------
# PASO 5: INGREDIENTES EN CASA
# ---------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("🛒 5. ¿Qué ingredientes tienes en casa?")
    
    ingredientes = st.multiselect(
        "Aprovecha lo que ya tienes en la despensa:",
        ["tomate", "cebolla", "ajo", "papa", "zanahoria", "leche", "arroz", "pasta", "mantequilla", "espinaca", "limon", "champiñones"],
        default=["tomate", "ajo", "cebolla"]
    )
    st.session_state["ingredientes"] = ingredientes

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 4
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 6
        st.rerun()

# ---------------------------------------------------------
# PASO 6: INGREDIENTES BLOQUEADOS Y ALERGIAS
# ---------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("🚫 6. Ingredientes bloqueados o alergias")
    
    excluidos = st.multiselect(
        "Omitir completamente recetas que contengan:",
        ["cerdo", "leche", "huevo", "pescado", "queso", "mantequilla", "papa"]
    )
    st.session_state["excluidos"] = excluidos

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 5
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True, type="primary"):
        st.session_state.step = 7
        st.rerun()

# ---------------------------------------------------------
# PASO 7: DÍAS Y CANTIDAD DE RECETAS
# ---------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("📅 7. ¿Cuántas recetas quieres planificar?")
    
    num_recetas = st.slider("Número de platillos a recomendar:", 1, 7, 5)
    st.session_state["num_recetas"] = num_recetas

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 6
        st.rerun()
    if col2.button("✨ Generar Menú Inteligente", use_container_width=True, type="primary"):
        st.session_state.step = 8
        st.rerun()

# ---------------------------------------------------------
# PASO 8: RESULTADOS Y MENÚ GENERADO
# ---------------------------------------------------------
elif st.session_state.step == 8:
    st.subheader("🎉 Tu Plan de Comidas Personalizado")
    
    p = st.session_state.get("personas", 2)
    pres = st.session_state.get("presupuesto", 60)
    st.info(f"📋 **Configuración:** {p} personas | Presupuesto aprox: {pres}€ / semana")

    df_filtered = df.copy()

    # Filtro de País
    pais_sel = st.session_state.get("pais", "Todos los países")
    if pais_sel != "Todos los países":
        df_filtered = df_filtered[df_filtered["cuisine"] == pais_sel]

    # Mapeo de ingredientes
    map_ing = {
        "tomate": "tomato", "pollo": "chicken", "ajo": "garlic", "cebolla": "onion",
        "carne": "beef", "cerdo": "pork", "queso": "cheese", "papa": "potato",
        "zanahoria": "carrot", "leche": "milk", "huevo": "egg", "arroz": "rice",
        "pasta": "pasta", "pescado": "fish", "mantequilla": "butter",
        "espinaca": "spinach", "limon": "lemon", "champiñones": "mushroom"
    }

    # Aplicar Exclusiones
    excluidos_cols = [map_ing[i] for i in st.session_state.get("excluidos", []) if i in map_ing]
    for col in excluidos_cols:
        if col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[col] == 0]

    # Aplicar Preferencias de Inclusión
    incluidos_cols = [map_ing[i] for i in st.session_state.get("ingredientes", []) + st.session_state.get("proteinas", []) if i in map_ing]
    
    if incluidos_cols:
        cols_existentes = [c for c in incluidos_cols if c in df_filtered.columns]
        if cols_existentes:
            df_filtered["match_score"] = df_filtered[cols_existentes].sum(axis=1)
            df_filtered = df_filtered.sort_values(by=["match_score", "rating"], ascending=False)

    resultados = df_filtered.head(st.session_state.get("num_recetas", 5))

    if len(resultados) == 0:
        st.warning("No encontramos recetas con esas restricciones estrictas. Prueba quitando algunos bloqueos.")
    else:
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
        
        for idx, (_, row) in enumerate(resultados.iterrows()):
            dia_txt = dias[idx % len(dias)]
            pais_txt = row.get("cuisine", "Internacional")
            
            # Formateo seguro para métricas
            cal = int(round(float(row.get('calories', 0))))
            prot = int(round(float(row.get('protein', 0))))
            gras = int(round(float(row.get('fat', 0))))
            rat = round(float(row.get('rating', 4.0)), 1)

            st.markdown(f"""
            <div class='recipe-card'>
                <span class='badge-country'>📍 Cocina {pais_txt}</span>
                <span class='badge-rating'>⭐ {rat} / 5.0</span>
                <h3 style='margin-top: 10px; color: #FFFFFF;'>🗓️ {dia_txt}: {row['title']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Calorías", f"{cal} kcal")
            c2.metric("Proteína", f"{prot} g")
            c3.metric("Grasas", f"{gras} g")

            if "steps" in row and pd.notna(row["steps"]) and str(row["steps"]).strip() != "":
                with st.expander("📖 Ver instrucciones paso a paso"):
                    st.text(row["steps"])
            st.markdown("---")

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Ajustar preferencias", use_container_width=True):
        st.session_state.step = 7
        st.rerun()
    if col2.button("🔄 Reiniciar planificador", use_container_width=True, type="primary"):
        st.session_state.step = 1
        st.rerun()