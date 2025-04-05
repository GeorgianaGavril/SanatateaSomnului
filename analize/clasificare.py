import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve
import streamlit as st


def conf_mtrx(y_test, y_pred, model_name):
    """
    Creează și afișează matricea de confuzie pentru un model.
    """
    # Calculează matricea de confuzie
    cm = confusion_matrix(y_test, y_pred)

    # Creează un plot cu dimensiune mai mică
    fig, ax = plt.subplots(figsize=(2, 2))

    # Creează heatmap pentru matricea de confuzie
    sns.heatmap(cm, annot=True, linewidths=0.2, linecolor="red", fmt=".0f", ax=ax)

    # Adaugă etichete pentru axe și titlu
    plt.xlabel("Valori prezise")
    plt.ylabel("Valori reale")
    plt.title(f"Matricea de confuzie - {model_name}", fontsize=2)  # Font mai mic pentru titlu

    # Ajustează marginile pentru a face figura mai compactă
    plt.tight_layout()

    return fig


def roc_auc_curve_plot(model, X_test, y_test):
    """
    Afișează curba ROC-AUC pentru un model de clasificare.
    """
    # Probabilitățile pentru modelul "No Skill"
    ns_probs = [0 for _ in range(len(y_test))]

    # Obținem probabilitățile pentru clasa pozitivă
    model_probs = model.predict_proba(X_test)[:, 1]

    # Calculăm scorurile AUC
    ns_auc = roc_auc_score(y_test, ns_probs)
    model_auc = roc_auc_score(y_test, model_probs)

    # Generăm datele pentru curba ROC
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

    # Creăm figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Desenăm curbele ROC
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label=f'No Skill (AUC = {ns_auc:.3f})')
    plt.plot(model_fpr, model_tpr, marker='.', label=f'Model (AUC = {model_auc:.3f})')

    # Adăugăm etichete și legendă
    plt.xlabel('Rată Fals Pozitivă')
    plt.ylabel('Rată Adevărat Pozitivă')
    plt.title('Curba ROC')
    plt.legend()

    return fig, model_auc


def page_analiza_clasificare(data):
    """
    Pagină pentru analiza de clasificare folosind regresia logistică.
    """
    st.title("Model de clasificare cu Regresie Logistică")

    st.markdown("""
    În această secțiune, vom aplica un model de clasificare (Regresie Logistică) pentru a prezice 
    o variabilă categorială din setul de date. Vom evalua performanța modelului folosind matricea 
    de confuzie, raportul de clasificare și curba ROC.
    """)

    if not isinstance(data, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return data

    if len(data) < 10:
        st.error("Nu există suficiente date pentru a crea un model de clasificare.")
        return data

    # Selectarea variabilei țintă pentru clasificare
    st.subheader("1. Selectarea variabilei țintă pentru clasificare")

    # Identificăm coloanele care pot fi folosite pentru clasificare (binare sau categoriale cu puține valori)
    potential_targets = []
    for col in data.columns:
        unique_vals = data[col].nunique()
        if unique_vals == 2:  # Variabile binare
            potential_targets.append(col)
        elif 2 < unique_vals <= 5:  # Variabile categoriale cu puține valori
            potential_targets.append(col)

    if not potential_targets:
        st.warning("Nu au fost găsite variabile potrivite pentru clasificare (binare sau cu puține categorii).")
        # Adaugă opțiunea de a selecta orice coloană
        potential_targets = data.columns.tolist()

    target_variable = st.selectbox(
        "Selectați variabila țintă pentru clasificare:",
        potential_targets
    )

    # Verifică dacă variabila țintă este binară
    unique_values = data[target_variable].nunique()

    if unique_values > 2:
        st.warning(
            f"Variabila '{target_variable}' are {unique_values} valori unice. Pentru simplitate, vom transforma aceasta într-o variabilă binară.")
        # Oferă opțiunea de a binariza variabila
        threshold = st.slider(
            f"Alegeți un prag pentru a binariza '{target_variable}':",
            float(data[target_variable].min()),
            float(data[target_variable].max()),
            float(data[target_variable].median())
        )

        # Binarizăm variabila
        y = (data[target_variable] > threshold).astype(int)
        st.info(
            f"Variabila a fost transformată: 0 = '{target_variable} <= {threshold}', 1 = '{target_variable} > {threshold}'")
    else:
        # Dacă variabila este deja binară, o folosim direct
        y = data[target_variable].astype(int)

    # Selectăm caracteristicile pentru model
    st.subheader("2. Selectarea caracteristicilor pentru model")

    # Obținem doar coloanele numerice
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Excludem variabila țintă
    numerical_columns = [col for col in numerical_columns if col != target_variable]

    selected_features = st.multiselect(
        "Selectați caracteristicile pentru model:",
        numerical_columns,
        default=numerical_columns[:min(5, len(numerical_columns))]  # Selectăm primele 5 coloane sau mai puține
    )

    if not selected_features:
        st.error("Trebuie să selectați cel puțin o caracteristică pentru model.")
        return data

    # Pregătirea datelor
    st.subheader("3. Împărțirea datelor în seturi de antrenare și testare")

    # Extragem datele
    X = data[selected_features]

    # Parametri pentru împărțirea datelor
    test_size = 0.2  # 20% pentru testare
    random_state = 42  # Pentru reproductibilitate

    # Divizăm datele
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    st.markdown(f"- Date pentru antrenare: {X_train.shape[0]} înregistrări ({(1 - test_size) * 100:.0f}%)")
    st.markdown(f"- Date pentru testare: {X_test.shape[0]} înregistrări ({test_size * 100:.0f}%)")

    # Antrenarea modelului
    st.subheader("4. Antrenarea modelului de Regresie Logistică")

    # Parametri pentru model
    max_iter = st.slider("Număr maxim de iterații:", 100, 1000, 100, 100)
    C = st.slider("Parametrul de regularizare (C):", 0.01, 10.0, 1.0, 0.1)

    if st.button("Antrenează modelul"):
        with st.spinner("Antrenez modelul de regresie logistică..."):
            # Creăm și antrenăm modelul
            model = LogisticRegression(max_iter=max_iter, C=C, random_state=random_state)
            model.fit(X_train, y_train)

            # Facem predicții
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Evaluarea modelului
            st.subheader("5. Evaluarea modelului")

            # Metrici de performanță
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            col1, col2 = st.columns(2)
            col1.metric("Acuratețe", f"{accuracy:.4f}")
            col2.metric("Scor F1", f"{f1:.4f}")

            # Matricea de confuzie
            st.markdown("### Matricea de confuzie")
            fig_cm = conf_mtrx(y_test, y_pred, "Regresie Logistică")
            st.pyplot(fig_cm)

            # Raportul de clasificare
            st.markdown("### Raportul de clasificare")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Curba ROC
            st.markdown("### Curba ROC")
            fig_roc, auc_score = roc_auc_curve_plot(model, X_test, y_test)
            st.pyplot(fig_roc)

            # Importanța caracteristicilor
            st.subheader("6. Importanța caracteristicilor")

            # Creăm un DataFrame cu coeficienții
            coefficients = pd.DataFrame({
                'Caracteristică': X.columns,
                'Coeficient': model.coef_[0]
            })
            coefficients['Importanță absolută'] = np.abs(coefficients['Coeficient'])
            coefficients = coefficients.sort_values('Importanță absolută', ascending=False)

            # Afișăm coeficienții
            st.dataframe(coefficients)

            # Vizualizăm coeficienții
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Coeficient', y='Caracteristică', data=coefficients)
            ax.set_title('Coeficienții modelului de regresie logistică')
            ax.axvline(x=0, color='gray', linestyle='--')
            st.pyplot(fig)

            # Interpretarea rezultatelor
            st.subheader("7. Interpretarea rezultatelor")

            st.markdown("""
            #### Metrici de performanță:
            - **Acuratețe**: Procentul de predicții corecte din totalul predicțiilor.
            - **Scor F1**: Media armonică între precizie și recall (sensibilitate).

            #### Matricea de confuzie:
            - **Adevărat Pozitiv (AP)**: Cazuri pozitive clasificate corect ca pozitive.
            - **Adevărat Negativ (AN)**: Cazuri negative clasificate corect ca negative.
            - **Fals Pozitiv (FP)**: Cazuri negative clasificate incorect ca pozitive.
            - **Fals Negativ (FN)**: Cazuri pozitive clasificate incorect ca negative.

            #### Curba ROC:
            - Arată performanța modelului la diferite praguri de clasificare.
            - **AUC (Area Under Curve)**: O valoare apropiată de 1 indică un model bun.

            #### Coeficienții modelului:
            - Coeficienți pozitivi indică o creștere a probabilității clasei pozitive.
            - Coeficienți negativi indică o scădere a probabilității clasei pozitive.
            - Valoarea absolută a coeficienților indică importanța caracteristicilor.
            """)

    return data