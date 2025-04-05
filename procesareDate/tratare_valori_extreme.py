import streamlit as st
from utils import *
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

def page_tratare_valori_extreme(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Tratarea valorilor extreme</h1>',
                unsafe_allow_html=True)

    # Obținem coloanele numerice
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Metoda IQR
    st.subheader("1. Metoda IQR")

    st.markdown(r"""
                Metoda Interquartile Range (IQR) este folosită pentru a detecta și elimina valorile extreme dintr-un set de date.
                - **Q1:** 25% din valorile cele mai mici
                - **Q3:** 75% din valorile cele mai mici
                - **IQR:** Q3 - Q1
                - **Outlieri:** valorile care sunt în afara intervalului acceptabil **[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]**""")

    # Calculate IQR and identify outliers for each numeric column
    for column in numeric_columns:
        st.markdown(f"#### Analiza outlierilor pentru: {column}")

        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

        st.write(f"Limita inferioară: {lower_bound}")
        st.write(f"Limita superioară: {upper_bound}")
        st.write(f"Număr de outlieri: {len(outliers)}")

        # Create boxplot to visualize outliers
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=data[column], ax=ax)
        ax.set_title(f'Boxplot pentru {column}')
        st.pyplot(fig)

        # Display the outliers if any exist
        if len(outliers) > 0:
            st.write("Valorile outlier:")
            st.write(outliers)
    st.markdown(
        "Singura variabilă cu valori extreme este Heart Rate. În general, un ritm cardiac de repaus mai mic este considerat normal. "
        "Observăm că persoanele respective au tulburări de somn, lucru ce poate contribui la un ritm cardiac mai ridicat. În plus, aceste persoane "
        "au meserii stresante și solicitante, precum doctor, avocat și programator. Așadar, vom elimina aceste valori.")

    # Remove all outliers from dataset
    set_date_1 = remove_outliers_iqr(data, numeric_columns)

    st.subheader("Setul de date după eliminarea outlierilor:")
    st.write(f"Număr de înregistrări după eliminarea outlierilor: {len(set_date_1)}")
    st.write(set_date_1)

    # Vizualizăm distribuția după eliminarea outlierilor
    n_cols = 3
    n_rows = math.ceil(len(numeric_columns) / n_cols)
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(set_date_1[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        plt.title(f'Distribuția (IQR): {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')

    plt.tight_layout()
    st.pyplot(plt)

    ##############################################################
    # 2. Metoda Z-Score
    st.subheader("2. Metoda Z-Score")
    st.markdown(
        "Metoda Z-Score determină cât de departe este o valoare de media setului de date, în unități de deviație standard. Se bazează pe formula:")
    st.latex(r"z = \frac{X - \mu}{\sigma}")
    st.markdown(r"""
                - **X** este valoarea dată
                - **μ** este media setului de date
                - **σ** este deviația standard""")
    st.markdown("Dacă |Z| > 3, se consideră outlier.")

    set_date_2 = remove_outliers_zscore(data, numeric_columns)

    # Afișăm noul set de date
    st.write(f"Număr de înregistrări după eliminarea outlierilor (Z-Score): {len(set_date_2)}")
    st.write(set_date_2)

    # Vizualizăm distribuția după Z-Score
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(set_date_2[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        plt.title(f'Distribuția (Z-Score): {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')

    plt.tight_layout()
    st.pyplot(plt)

    ##########################################################
    # Metoda 3. Transformarea logaritmica a datelor
    st.subheader("3. Metoda transformării logaritmice a datelor")

    st.markdown(
        "Logartimarea reduce asimetria distribuției și efectul valorilor mari. Mai întâi trebuie să ne asigurăm că datele sunt strict pozitive, "
        "deoarece logaritmul nu este definit pentru valori negative sau zero.")

    # Verificăm existența valorilor zero sau negative
    numeric_columns_new = [col for col in numeric_columns if not (data[col] <= 0).any()]

    for col in numeric_columns:
        if col not in numeric_columns_new:
            st.warning(f"Coloana {col} conține valori zero sau negative, deci nu poate fi logaritmată.")

    # Aplicăm log(1 + x)
    set_date_3 = apply_log_transform(data, numeric_columns_new)

    st.write("Setul de date după aplicarea transformării logaritmice:")
    st.write(set_date_3)

    # Vizualizăm distribuția după transformarea logaritmică
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(set_date_3[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        plt.title(f'Distribuția (Log): {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')
    plt.tight_layout()
    st.pyplot(plt)

    st.markdown("Observăm că cele 3 transformări nu aduc modificări majore asupra setului de date. "
                "Mai departe vom continua cu setul de date rezultat din metoda IQR.")

    # Selectarea metodei de tratare a valorilor extreme
    method_choice = st.selectbox(
        "Alegeți metoda de tratare a valorilor extreme pentru etapele următoare:",
        ["IQR", "Z-Score", "Transformare logaritmică"]
    )

    if method_choice == "IQR":
        st.success("Ați ales metoda IQR. Datele tratate cu această metodă vor fi folosite pentru etapele următoare.")
        final_cleaned_data = set_date_1
    elif method_choice == "Z-Score":
        st.success(
            "Ați ales metoda Z-Score. Datele tratate cu această metodă vor fi folosite pentru etapele următoare.")
        final_cleaned_data = set_date_2
    else:
        st.success(
            "Ați ales metoda de transformare logaritmică. Datele tratate cu această metodă vor fi folosite pentru etapele următoare.")
        final_cleaned_data = set_date_3

    if st.button("Salvează setul de date după tratarea valorilor extreme"):
        final_cleaned_data.to_csv("sleep_health_cleaned.csv")
        st.success("Datele au fost salvate cu succes!")

    return final_cleaned_data