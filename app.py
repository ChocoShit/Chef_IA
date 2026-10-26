import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Chef IA - Planificador Inteligente", page_icon="🍳", layout="centered")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("epi_r_combinado.csv")
    except FileNotFoundError:
        df = pd.read_csv("epi_r.csv")
    return df

df = load_data()

if "step" not in st.session_state:
    st.session_state.step = 1

st.title("🍳 Chef IA — Planificador de Menú")
st.caption("Configura tus preferencias paso a paso y la IA generará las mejores recetas para ti.")

progress_val = int((st.session_state.step / 4) * 100)
st.progress(progress_val)
st.write(f"**Paso {st.session_state.step} de 4**")

st.markdown("---")

if st.session_state.step == 1:
    st.subheader("1️⃣ ¿Qué origen o cocina prefieres hoy?")
    
    opciones_cocina = ["Cualquiera"]
    if "cuisine" in df.columns:
        opciones_cocina += list(df["cuisine"].value_counts().index)
    
    cocina = st.selectbox("Selecciona el país/origen de la comida:", opciones_cocina)
    st.session_state["cocina"] = cocina

    st.subheader("¿Para qué momento del día?")
    momento = st.radio("Momento:", ["Cualquiera", "Desayuno", "Almuerzo/Cena", "Postre", "Snack"], horizontal=True)
    st.session_state["momento"] = momento

    if st.button("Siguiente ➡️", use_container_width=True):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.subheader("2️⃣ Ingredientes y Proteínas")
    
    proteinas = st.multiselect(
        "¿Qué fuentes de proteína prefieres incluir?",
        ["pollo", "carne", "cerdo", "pescado", "huevo", "queso"],
        default=["pollo"]
    )
    st.session_state["proteinas"] = proteinas

    ingredientes = st.multiselect(
        "Ingredientes adicionales que tienes en casa:",
        ["tomate", "cebolla", "ajo", "papa", "zanahoria", "leche", "arroz", "pasta", "mantequilla"],
        default=["tomate", "ajo"]
    )
    st.session_state["ingredientes"] = ingredientes

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Siguiente ➡️", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

elif st.session_state.step == 3:
    st.subheader("3️⃣ Ingredientes Bloqueados y Alergias")
    
    excluidos = st.multiselect(
        "Selecciona ingredientes que quieras excluir por completo:",
        ["cerdo", "leche", "huevo", "pescado", "queso", "mantequilla", "papa"]
    )
    st.session_state["excluidos"] = excluidos

    num_recetas = st.slider("¿Cuántas opciones de menú quieres generar?", 1, 10, 3)
    st.session_state["num_recetas"] = num_recetas

    col1, col2 = st.columns(2)
    if col1.button("⬅️ Atrás", use_container_width=True):
        st.session_state.step = 2
        st.rerun()
    if col2.button("✨ Generar Menú con IA", use_container_width=True, type="primary"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.subheader("🎉 Tu Menú Personalizado con IA")

    df_filtered = df.copy()

    if st.session_state.get("cocina") and st.session_state["cocina"] != "Cualquiera":
        if "cuisine" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["cuisine"] == st.session_state["cocina"]]

    map_ing = {
        "tomate": "tomato", "pollo": "chicken", "ajo": "garlic", "cebolla": "onion",
        "carne": "beef", "cerdo": "pork", "queso": "cheese", "papa": "potato",
        "zanahoria": "carrot", "leche": "milk", "huevo": "egg", "arroz": "rice",
        "pasta": "pasta", "pescado": "fish", "mantequilla": "butter"
    }

    excluidos_cols = [map_ing[i] for i in st.session_state.get("excluidos", []) if i in map_ing]
    for col in excluidos_cols:
        if col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[col] == 0]

    incluidos_cols = [map_ing[i] for i in st.session_state.get("ingredientes", []) + st.session_state.get("proteinas", []) if i in map_ing]
    
    if incluidos_cols:
        cols_existentes = [c for c in incluidos_cols if c in df_filtered.columns]
        if cols_existentes:
            df_filtered["match_score"] = df_filtered[cols_existentes].sum(axis=1)
            df_filtered = df_filtered.sort_values(by=["match_score", "rating"], ascending=False)

    resultados = df_filtered.head(st.session_state.get("num_recetas", 3))

    if len(resultados) == 0:
        st.warning("No encontramos recetas con esa combinación de restricciones. Intenta quitar algunos bloqueos.")
    else:
        for idx, row in resultados.iterrows():
            pais_txt = row.get("cuisine", "Internacional")
            st.markdown(f"### 🍲 {row['title']} — *📍 Cocina {pais_txt}*")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Calorías", f"{int(row.get('calories', 0))} kcal")
            c2.metric("Proteína", f"{int(row.get('protein', 0))} g")
            c3.metric("Grasas", f"{int(row.get('fat', 0))} g")

            if "steps" in row and pd.notna(row["steps"]):
                with st.expander("📖 Ver modo de preparación paso a paso"):
                    st.text(row["steps"])
            st.markdown("---")

    if st.button("🔄 Crear Nuevo Plan Desde Cero", use_container_width=True):
        st.session_state.step = 1
        st.rerun()