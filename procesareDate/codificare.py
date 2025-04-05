import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def page_codificare_date(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Codificarea datelor categorice</h1>',
                unsafe_allow_html=True)

    # Create a copy of the dataset for encoding
    encoded_data = data.copy()

    # Identify categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    st.write(f"Coloane categorice identificate: {categorical_columns}")

    # 1. Label Encoding (pentru variabile ordinale sau binare)
    st.markdown("### 1. Label Encoding")
    st.markdown(
        "Această metodă înlocuiește fiecare categorie cu un număr întreg. Este potrivită pentru variabile ordinale sau binare.")

    label_encoder = LabelEncoder()
    le_data = encoded_data.copy()

    for column in ['Gender', 'Sleep Disorder']:
        le_data[f'{column}_Label'] = label_encoder.fit_transform(le_data[column])
        # Afișează maparea pentru claritate
        mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        st.write(f"Mapare pentru {column}: {mapping}")

    st.write("Date după Label Encoding:")
    st.write(le_data)

    # 2. One-Hot Encoding (pentru variabile nominale)
    st.markdown("### 2. One-Hot Encoding")
    st.markdown(
        "Această metodă creează coloane noi pentru fiecare categorie. Este potrivită pentru variabile nominale fără ordine specifică.")

    ohe_data = encoded_data.copy()
    for column in ['BMI Category', 'Occupation']:
        # Crearea dummies și excluderea primei coloane pentru a evita colinearitatea
        dummies = pd.get_dummies(ohe_data[column], prefix=column, drop_first=True)
        # Adăugarea dummy variabilelor la dataset
        ohe_data = pd.concat([ohe_data, dummies], axis=1)

    st.write("Date după One-Hot Encoding pentru BMI Category și Occupation:")
    st.write(ohe_data)

    # 3. Encoding pentru Blood Pressure (separare în sistolic și diastolic)
    st.markdown("### 3. Encoding special pentru Blood Pressure")
    st.markdown("Separăm valorile tensiunii arteriale în valori sistolice și diastolice.")

    if 'Blood Pressure' in encoded_data.columns:
        bp_data = encoded_data.copy()
        # Extrage valorile sistolice și diastolice din formatul "120/80"
        bp_data['Systolic'] = bp_data['Blood Pressure'].str.split('/').str[0].astype(int)
        bp_data['Diastolic'] = bp_data['Blood Pressure'].str.split('/').str[1].astype(int)

        st.write("Date după separarea Blood Pressure:")
        st.write(bp_data)

    # 4. Combinarea metodelor pentru un dataset final pregătit pentru analiză
    st.markdown("### 4. Dataset final codificat")
    st.markdown(
        "Combinăm cele mai potrivite metode de codificare pentru a crea un dataset final pregătit pentru analiză.")

    final_data = encoded_data.copy()

    # Label Encoding pentru variabile binare
    for column in ['Gender', 'Sleep Disorder']:
        final_data[f'{column}_Encoded'] = label_encoder.fit_transform(final_data[column])

    # One-Hot Encoding pentru variabile nominale cu mai multe categorii
    for column in ['BMI Category', 'Occupation']:
        dummies = pd.get_dummies(final_data[column], prefix=column, drop_first=True)
        final_data = pd.concat([final_data, dummies], axis=1)

    # Procesare Blood Pressure
    if 'Blood Pressure' in final_data.columns:
        final_data['Systolic'] = final_data['Blood Pressure'].str.split('/').str[0].astype(int)
        final_data['Diastolic'] = final_data['Blood Pressure'].str.split('/').str[1].astype(int)

    # Eliminăm coloanele originale care au fost codificate
    final_data_clean = final_data.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder', 'Blood Pressure'],
                                       axis=1)

    st.write("Dataset final codificat (cu coloanele originale înlocuite):")
    st.write(final_data_clean)

    # Salvarea datelor pentru analiză ulterioară
    st.markdown("### Salvarea datelor codificate")
    st.markdown("Datele pot fi salvate pentru analiză ulterioară.")

    if st.button('Salvează datele codificate ca CSV'):
        final_data_clean.to_csv('sleep_health_encoded.csv')
        st.success('Datele au fost salvate cu succes!')

    return final_data_clean
