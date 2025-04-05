import streamlit as st
from procesareDate.tratare_valori_extreme import *
from procesareDate.codificare import *
from procesareDate.scalare import *
from procesareDate.prezentare_generala import *
from analize.analiza_distributiei import *
from procesareDate.tratare_valori_duplicat_nule import *
from analize.model_regresie import page_model_regresie
from analize.analiza_cluster import page_analiza_clustering
from analize.clasificare import page_analiza_clasificare

# Configurare pagină
st.set_page_config(page_title="Sănătatea somnului", page_icon="💤", layout="wide")

# Funcția pentru încărcarea și pregătirea datelor
def load_data():
    data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", index_col=0)
    return data

def main():
    # Sidebar
    st.sidebar.title("Navigare")
    pages = [
        "Prezentare generală",
        "Tratarea valorilor duplicat si nule",
        "Analiza distribuției datelor",
        "Tratarea valorilor extreme",
        "Codificarea datelor",
        "Scalarea datelor",
        "Model de regresie",
        "Analiza clustering",
        "Model de clasificare"  # Am adăugat pagina pentru clasificare
    ]
    page = st.sidebar.radio("Selectați secțiunea:", pages)

    with st.sidebar:
        st.markdown("---")
        st.markdown("## Despre aplicație")
        st.markdown("Această aplicație permite analiza unui set de date despre sănătatea somnului.")

        # Adăugăm buton pentru descărcarea datelor originale
        st.markdown("---")
        st.markdown("### Export date")

        if 'data' in st.session_state:
            if st.button("Descarcă setul original"):
                csv = st.session_state.data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Descarcă CSV",
                    data=csv,
                    file_name="sleep_health_original.csv",
                    mime="text/csv"
                )

        # Stilizare sidebar
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

    # Inițializăm starea sesiunii dacă nu există
    if 'data' not in st.session_state:
        st.session_state.data = load_data()

    if page == "Prezentare generală":
        st.session_state.data = page_prezentare_generala(st.session_state.data)
    elif page == "Tratarea valorilor duplicat si nule":
        st.session_state.data = page_tratare_valori_duplicate_nule(st.session_state.data)
    elif page == "Analiza distribuției datelor":
        st.session_state.data = page_analiza_distributie(st.session_state.data)
    elif page == "Tratarea valorilor extreme":
        st.session_state.data = page_tratare_valori_extreme(st.session_state.data)
    elif page == "Codificarea datelor":
        st.session_state.data = page_codificare_date(st.session_state.data)
    elif page == "Scalarea datelor":
        st.session_state.data = page_scalare_date(st.session_state.data)
    elif page == "Analiza clustering":
        st.session_state.data = page_analiza_clustering(st.session_state.data)
    elif page == "Model de regresie":
        st.session_state.data = page_model_regresie(st.session_state.data)
    elif page == "Model de clasificare":
        st.session_state.data = page_analiza_clasificare(st.session_state.data)

if __name__ == "__main__":
    main()