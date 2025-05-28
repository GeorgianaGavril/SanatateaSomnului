import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st


def page_model_regresie(data):
    st.title("Model de regresie liniară pentru predicția calității somnului")

    st.markdown("""
    În această secțiune, vom construi un model de regresie liniară pentru a prezice calitatea somnului
    bazată pe celelalte caracteristici disponibile în setul de date. Am ales "Quality of Sleep" ca variabilă țintă
    deoarece este un indicator esențial al sănătății somnului și este influențat de diverși factori precum
    stresul, durata somnului, activitatea fizică etc. Înțelegerea acestora poate oferi informații valoroase 
    pentru îmbunătățirea sănătății generale.
    """)

    if not isinstance(data, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return data

    # Definim direct variabila țintă ca fiind "Quality of Sleep"
    target_variable = "Quality of Sleep"

    if target_variable not in data.columns:
        st.error(f"Variabila țintă '{target_variable}' nu există în setul de date.")
        return data

    st.subheader("1. Separarea datelor în caracteristici de intrare și țintă")

    st.markdown(f"""
    Pregătim datele pentru antrenarea modelului de regresie:
    - Variabila țintă: **{target_variable}**
    - Caracteristici de intrare: toate celelalte coloane relevante
    """)

    # Verificăm dacă există suficiente date pentru a crea un model
    if len(data) < 10:
        st.error("Nu există suficiente date pentru a crea un model de regresie.")
        return data

    # Separăm variabila țintă și variabilele de intrare
    y = data[target_variable].values

    # Verificăm dacă valorile țintei sunt foarte mari și aplicăm logaritmul dacă este cazul
    apply_log = False
    if y.max() > 1000 or y.var() > 10000:
        apply_log = st.checkbox("Aplică transformare logaritmică pentru variabila țintă", value=True)
        if apply_log:
            y = np.log(y + 1)  # Adăugăm 1 pentru a evita log(0)
            st.info("Am aplicat logaritmul natural pentru variabila țintă pentru a îmbunătăți performanța modelului.")

    # Selectăm caracteristicile pentru predicție (eliminăm variabila țintă)
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    X_columns = st.multiselect(
        "Selectați caracteristicile pentru model:",
        [col for col in data.columns if col != target_variable],
        default=[col for col in numerical_columns if col != target_variable]
    )

    if not X_columns:
        st.error("Trebuie să selectați cel puțin o caracteristică pentru model.")
        return data

    X = data[X_columns]

    # Verificăm dacă avem coloane categorice și le codificăm
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    if not categorical_columns.empty:
        st.subheader("2. Codificarea variabilelor categorice")
        st.markdown("Următoarele variabile categorice vor fi codificate folosind one-hot encoding:")
        st.write(categorical_columns.tolist())

        # Aplicăm one-hot encoding pentru variabilele categorice
        X = pd.get_dummies(X)
        st.success(f"Codificare realizată cu succes. Noul set de date are {X.shape[1]} caracteristici.")

    # Împărțirea datelor în seturi de antrenare și testare
    st.subheader("2. Împărțirea datelor în seturi de antrenare și testare")

    st.markdown("""
    Urmează să împărțim setul de date în două subseturi: unul pentru antrenarea modelului și altul pentru testare. 
    Această separare este esențială în procesul de învățare automată, deoarece ne permite să evaluăm corect performanța modelului.

    Setul de antrenare permite modelului să învețe relațiile dintre variabile și să identifice tipare,
    în timp ce setul de testare ne ajută să verificăm cât de bine generalizează modelul pe date noi, nevăzute în timpul antrenării.
    Vom folosi 80% din date pentru antrenare și 20% pentru testare, un raport standard în domeniul învățării automate.
    """)

    # Folosim valori fixe pentru împărțire: 20% test, random_state=42
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.markdown(f"- Date pentru antrenare: {X_train.shape[0]} înregistrări ({(1 - test_size) * 100:.0f}%)")
    st.markdown(f"- Date pentru testare: {X_test.shape[0]} înregistrări ({test_size * 100:.0f}%)")

    # Antrenarea modelului de regresie liniară
    st.subheader("3. Antrenarea modelului de regresie liniară")

    # Verificăm dacă modelul există deja în session_state pentru a evita reantrenarea
    train_model = False
    if 'lr_model' not in st.session_state or st.button("Antrenează modelul"):
        train_model = True

    # Dacă trebuie să antrenăm modelul sau dacă utilizatorul a apăsat butonul
    if train_model:
        with st.spinner("Antrenez modelul de regresie liniară..."):
            # Construirea și antrenarea modelului
            lr = linear_model.LinearRegression()
            model = lr.fit(X_train, y_train)

            # Salvarea modelului și a datelor relevante în session_state
            st.session_state.lr_model = model
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.target_variable = target_variable
            st.session_state.apply_log = apply_log

            # Predicțiile pe setul de antrenare
            y_train_pred = model.predict(X_train)

            # Afișarea rezultatelor pe setul de antrenare
            st.subheader("4. Evaluarea modelului pe setul de antrenare")

            # Crearea graficului de comparație pentru setul de antrenare
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_train_pred, y_train, alpha=0.5)
            ax.set_xlabel('Valori prezise')
            ax.set_ylabel('Valori reale')
            ax.set_title('Comparație între valorile reale și cele prezise (set de antrenare)')

            # Adăugăm linia ideală (y=x)
            min_val = min(y_train.min(), y_train_pred.min())
            max_val = max(y_train.max(), y_train_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')

            st.pyplot(fig)

            # Calcularea metricilor de performanță pentru setul de antrenare
            mse_train = mean_squared_error(y_train, y_train_pred)
            rmse_train = np.sqrt(mse_train)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            st.session_state.train_metrics = {
                'mse': mse_train,
                'rmse': rmse_train,
                'mae': mae_train,
                'r2': r2_train
            }

            st.markdown("**Metrici de performanță (set de antrenare):**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse_train:.4f}")
            col2.metric("RMSE", f"{rmse_train:.4f}")
            col3.metric("MAE", f"{mae_train:.4f}")
            col4.metric("R²", f"{r2_train:.4f}")

            # Predicțiile pe setul de testare
            y_test_pred = model.predict(X_test)

            # Afișarea rezultatelor pe setul de testare
            st.subheader("5. Evaluarea modelului pe setul de testare")

            # Crearea graficului de comparație pentru setul de testare
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(y_test_pred, y_test, alpha=0.5)
            ax.set_xlabel('Valori prezise')
            ax.set_ylabel('Valori reale')
            ax.set_title('Comparație între valorile reale și cele prezise (set de testare)')

            # Adăugăm linia ideală (y=x)
            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r-')

            st.pyplot(fig)

            # Calcularea metricilor de performanță pentru setul de testare
            mse_test = mean_squared_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mse_test)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            st.session_state.test_metrics = {
                'mse': mse_test,
                'rmse': rmse_test,
                'mae': mae_test,
                'r2': r2_test
            }

            st.markdown("**Metrici de performanță (set de testare):**")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", f"{mse_test:.4f}")
            col2.metric("RMSE", f"{rmse_test:.4f}")
            col3.metric("MAE", f"{mae_test:.4f}")
            col4.metric("R²", f"{r2_test:.4f}")

            # Explicarea metricilor și interpretarea rezultatelor
            with st.expander("Explicarea metricilor și interpretarea rezultatelor"):
                st.markdown("""
                ### Metrici de performanță:

                - **MSE (Mean Squared Error)**: Media pătratelor diferențelor dintre valorile reale și cele prezise.
                  - *Interpretare*: Valori mai mici indică o potrivire mai bună a modelului. MSE penalizează erorile mari mai sever decât cele mici.
                  - *Limită*: Nu are o scală fixă, depinde de datele analizate.

                - **RMSE (Root Mean Squared Error)**: Rădăcina pătrată a MSE. Este în aceleași unități ca variabila țintă.
                  - *Interpretare*: Reprezintă aproximativ "eroarea medie" în unitatea variabilei țintă. Un RMSE de 0.5 pentru calitatea somnului înseamnă că, în medie, predicțiile sunt cu ±0.5 puncte diferite de valorile reale.
                  - *Când este bun*: RMSE < 10% din intervalul valorilor variabilei țintă este considerat bun.

                - **MAE (Mean Absolute Error)**: Media valorilor absolute ale erorilor.
                  - *Interpretare*: Similar cu RMSE, dar tratează toate erorile în mod egal, indiferent de mărime.
                  - *Comparație cu RMSE*: Dacă MAE este semnificativ mai mic decât RMSE, înseamnă că există câteva erori mari ("outliers" în predicții).

                - **R² (Coeficient de determinare)**: Procentul din variația variabilei țintă explicat de model.
                  - *Interpretare*: 
                     - R² = 1.0: Modelul explică perfect variația (predicție perfectă)
                     - R² = 0.7: Modelul explică 70% din variația datelor
                     - R² = 0: Modelul nu este mai bun decât media simplă a datelor
                     - R² < 0: Modelul este mai rău decât media simplă (extrem de slab)
                  - *Evaluare calitativă*:
                     - R² > 0.9: Excelent
                     - R² între 0.7-0.9: Bun
                     - R² între 0.5-0.7: Moderat
                     - R² între 0.3-0.5: Slab
                     - R² < 0.3: Foarte slab

                ### Interpretarea diferenței între setul de antrenare și testare:

                - **Performanță similară**: Dacă metricile sunt apropiate pe ambele seturi, modelul generalizează bine.
                - **Performanță mult mai bună pe antrenare**: Indică supraadjustare (overfitting) - modelul "memorează" datele de antrenare dar nu generalizează.
                - **Performanță mult mai slabă pe testare**: Poate indica că setul de testare conține tipuri de date diferite față de cel de antrenare.

                ### Interpretarea graficelor:

                - **Puncte aproape de linia roșie**: Predicții bune
                - **Puncte răspândite aleatoriu**: Model slab
                - **Puncte formând un tipar (curba, grupuri)**: Relație nelineară care nu e capturată de model
                - **Puncte mai depărtate la capete**: Model care nu captează bine valorile extreme
                """)

            # Creăm un DataFrame cu coeficienții modelului
            coefficients = pd.DataFrame({
                'Caracteristică': X.columns,
                'Coeficient': model.coef_
            }).sort_values(by='Coeficient', ascending=False)

            # Calculează valorile absolute pentru a evalua importanța indiferent de direcție
            coefficients['Importanță (|Coeficient|)'] = np.abs(coefficients['Coeficient'])
            coefficients_abs = coefficients.sort_values(by='Importanță (|Coeficient|)', ascending=False)

            # Salvăm coeficienții pentru reutilizare
            st.session_state.coefficients = coefficients
            st.session_state.coefficients_abs = coefficients_abs

            # Importanța caracteristicilor
            st.subheader("6. Importanța caracteristicilor")

            # Afișăm un grafic cu importanța caracteristicilor (după valoarea absolută)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x='Importanță (|Coeficient|)', y='Caracteristică', data=coefficients_abs.head(15), ax=ax)
            ax.set_title('Top caracteristici după importanță absolută')
            st.pyplot(fig)

            # Afișăm un grafic cu coeficienții (cu semn) pentru a vedea direcția influenței
            fig, ax = plt.subplots(figsize=(10, 8))
            top_pos_coeffs = coefficients.head(10)
            top_neg_coeffs = coefficients.tail(10)
            top_coeffs = pd.concat([top_pos_coeffs, top_neg_coeffs]).sort_values(by='Coeficient')
            sns.barplot(x='Coeficient', y='Caracteristică', data=top_coeffs, ax=ax)
            ax.set_title('Top 10 coeficienți pozitivi și negativi')
            ax.axvline(x=0, color='gray', linestyle='--')
            st.pyplot(fig)

            # Afișăm coeficienții în format tabelar
            st.markdown("**Coeficienții modelului:**")
            st.dataframe(coefficients_abs)

            # Interpretarea coeficienților
            with st.expander("Interpretarea coeficienților"):
                st.markdown("""
                ### Cum să interpretezi coeficienții modelului:

                #### Magnitudinea coeficienților (importanța)
                - **Coeficienți cu valoare absolută mare**: Caracteristicile cu valorile cele mai mari (pozitive sau negative) au cel mai mare impact asupra predicției.
                - **Coeficienți aproape de zero**: Aceste caracteristici au o influență minimă în model.

                #### Direcția coeficienților (semnul)
                - **Coeficienți pozitivi**: O creștere a acestei caracteristici conduce la o creștere a calității somnului. De exemplu, dacă "Durata somnului" are un coeficient pozitiv, atunci cu cât doarme cineva mai mult, cu atât calitatea somnului este mai bună (conform modelului).
                - **Coeficienți negativi**: O creștere a acestei caracteristici conduce la o scădere a calității somnului. De exemplu, dacă "Nivelul de stres" are un coeficient negativ, atunci cu cât stresul este mai mare, cu atât calitatea somnului este mai scăzută.

                #### Atenție la interpretare:
                1. **Unități diferite**: Coeficienții reflectă și unitățile de măsură ale caracteristicilor. O caracteristică măsurată în mii va avea un coeficient mai mic decât una măsurată în unități, chiar dacă importanța reală este similară.
                2. **Colinearitate**: Dacă există caracteristici puternic corelate, interpretarea individuală a coeficienților poate fi înșelătoare.
                3. **Variabile dummy**: Pentru variabilele categorice codificate (one-hot encoding), coeficienții arată diferența față de categoria de referință.

                #### Exemplu de interpretare:
                Dacă "Durata somnului" are un coeficient de 0.5, înseamnă că o creștere de 1 unitate (de exemplu, o oră) în durata somnului este asociată cu o creștere de 0.5 unități în calitatea somnului (presupunând că toate celelalte variabile rămân constante).
                """)

                # Adăugăm o interpretare specifică pentru top 3 coeficienți (pozitivi și negativi)
                st.markdown("### Interpretarea specifică a celor mai influente caracteristici:")

                top_pos = coefficients.head(3)
                top_neg = coefficients.tail(3).iloc[::-1]

                st.markdown("#### Caracteristici cu influență pozitivă:")
                for i, row in top_pos.iterrows():
                    st.markdown(
                        f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o creștere de {row['Coeficient']:.4f} unități în calitatea somnului, menținând toate celelalte variabile constante.")

                st.markdown("#### Caracteristici cu influență negativă:")
                for i, row in top_neg.iterrows():
                    st.markdown(
                        f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o scădere de {abs(row['Coeficient']):.4f} unități în calitatea somnului, menținând toate celelalte variabile constante.")

    # Afișăm informațiile salvate dacă modelul a fost antrenat anterior
    elif 'lr_model' in st.session_state:
        # Afișarea rezultatelor pe setul de antrenare
        st.subheader("4. Evaluarea modelului pe setul de antrenare")

        st.markdown("**Metrici de performanță (set de antrenare):**")
        train_metrics = st.session_state.train_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{train_metrics['mse']:.4f}")
        col2.metric("RMSE", f"{train_metrics['rmse']:.4f}")
        col3.metric("MAE", f"{train_metrics['mae']:.4f}")
        col4.metric("R²", f"{train_metrics['r2']:.4f}")

        # Afișarea rezultatelor pe setul de testare
        st.subheader("4. Evaluarea modelului pe setul de testare")

        st.markdown("**Metrici de performanță (set de testare):**")
        test_metrics = st.session_state.test_metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{test_metrics['mse']:.4f}")
        col2.metric("RMSE", f"{test_metrics['rmse']:.4f}")
        col3.metric("MAE", f"{test_metrics['mae']:.4f}")
        col4.metric("R²", f"{test_metrics['r2']:.4f}")

        # Explicarea metricilor
        with st.expander("Explicarea metricilor și interpretarea rezultatelor"):
            st.markdown("""
            ### Metrici de performanță:

            - **MSE (Mean Squared Error)**: Media pătratelor diferențelor dintre valorile reale și cele prezise.
              - *Interpretare*: Valori mai mici indică o potrivire mai bună a modelului. MSE penalizează erorile mari mai sever decât cele mici.
              - *Limită*: Nu are o scală fixă, depinde de datele analizate.

            - **RMSE (Root Mean Squared Error)**: Rădăcina pătrată a MSE. Este în aceleași unități ca variabila țintă.
              - *Interpretare*: Reprezintă aproximativ "eroarea medie" în unitatea variabilei țintă. Un RMSE de 0.5 pentru calitatea somnului înseamnă că, în medie, predicțiile sunt cu ±0.5 puncte diferite de valorile reale.
              - *Când este bun*: RMSE < 10% din intervalul valorilor variabilei țintă este considerat bun.

            - **MAE (Mean Absolute Error)**: Media valorilor absolute ale erorilor.
              - *Interpretare*: Similar cu RMSE, dar tratează toate erorile în mod egal, indiferent de mărime.
              - *Comparație cu RMSE*: Dacă MAE este semnificativ mai mic decât RMSE, înseamnă că există câteva erori mari ("outliers" în predicții).

            - **R² (Coeficient de determinare)**: Procentul din variația variabilei țintă explicat de model.
              - *Interpretare*: 
                 - R² = 1.0: Modelul explică perfect variația (predicție perfectă)
                 - R² = 0.7: Modelul explică 70% din variația datelor
                 - R² = 0: Modelul nu este mai bun decât media simplă a datelor
                 - R² < 0: Modelul este mai rău decât media simplă (extrem de slab)
              - *Evaluare calitativă*:
                 - R² > 0.9: Excelent
                 - R² între 0.7-0.9: Bun
                 - R² între 0.5-0.7: Moderat
                 - R² între 0.3-0.5: Slab
                 - R² < 0.3: Foarte slab

            ### Interpretarea diferenței între setul de antrenare și testare:

            - **Performanță similară**: Dacă metricile sunt apropiate pe ambele seturi, modelul generalizează bine.
            - **Performanță mult mai bună pe antrenare**: Indică supraadjustare (overfitting) - modelul "memorează" datele de antrenare dar nu generalizează.
            - **Performanță mult mai slabă pe testare**: Poate indica că setul de testare conține tipuri de date diferite față de cel de antrenare.

            ### Interpretarea graficelor:

            - **Puncte aproape de linia roșie**: Predicții bune
            - **Puncte răspândite aleatoriu**: Model slab
            - **Puncte formând un tipar (curba, grupuri)**: Relație nelineară care nu e capturată de model
            - **Puncte mai depărtate la capete**: Model care nu captează bine valorile extreme
            """)

        # Importanța caracteristicilor
        st.subheader("5. Importanța caracteristicilor")

        # Afișăm coeficienții în format tabelar
        st.markdown("**Coeficienții modelului:**")
        st.dataframe(st.session_state.coefficients_abs)

        # Afișăm un grafic cu coeficienții (cu semn) pentru a vedea direcția influenței
        fig, ax = plt.subplots(figsize=(10, 8))
        coefficients = st.session_state.coefficients
        top_pos_coeffs = coefficients.head(10)
        top_neg_coeffs = coefficients.tail(10)
        top_coeffs = pd.concat([top_pos_coeffs, top_neg_coeffs]).sort_values(by='Coeficient')
        sns.barplot(x='Coeficient', y='Caracteristică', data=top_coeffs, ax=ax)
        ax.set_title('Top coeficienți pozitivi și negativi')
        ax.axvline(x=0, color='gray', linestyle='--')
        st.pyplot(fig)

        # Interpretarea coeficienților
        with st.expander("Interpretarea coeficienților"):
            st.markdown("""
            ### Cum să interpretezi coeficienții modelului:

            #### Magnitudinea coeficienților (importanța)
            - **Coeficienți cu valoare absolută mare**: Caracteristicile cu valorile cele mai mari (pozitive sau negative) au cel mai mare impact asupra predicției.
            - **Coeficienți aproape de zero**: Aceste caracteristici au o influență minimă în model.

            #### Direcția coeficienților (semnul)
            - **Coeficienți pozitivi**: O creștere a acestei caracteristici conduce la o creștere a calității somnului. De exemplu, dacă "Durata somnului" are un coeficient pozitiv, atunci cu cât doarme cineva mai mult, cu atât calitatea somnului este mai bună (conform modelului).
            - **Coeficienți negativi**: O creștere a acestei caracteristici conduce la o scădere a calității somnului. De exemplu, dacă "Nivelul de stres" are un coeficient negativ, atunci cu cât stresul este mai mare, cu atât calitatea somnului este mai scăzută.
            """)

            # Adăugăm o interpretare specifică pentru top 3 coeficienți (pozitivi și negativi)
            st.markdown("### Interpretarea specifică a celor mai influente caracteristici:")

            top_pos = st.session_state.coefficients.head(3)
            top_neg = st.session_state.coefficients.tail(3).iloc[::-1]

            st.markdown("#### Caracteristici cu influență pozitivă:")
            for i, row in top_pos.iterrows():
                st.markdown(
                    f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o creștere de {row['Coeficient']:.4f} unități în calitatea somnului, menținând toate celelalte variabile constante.")

            st.markdown("#### Caracteristici cu influență negativă:")
            for i, row in top_neg.iterrows():
                st.markdown(
                    f"- **{row['Caracteristică']}** (coeficient: {row['Coeficient']:.4f}): O creștere de 1 unitate în această caracteristică este asociată cu o scădere de {abs(row['Coeficient']):.4f} unități în calitatea somnului, menținând toate celelalte variabile constante.")

    # Secțiunea pentru predicția individuală
    if 'lr_model' in st.session_state:
        st.subheader("7. Predicție pentru un singur caz")

        st.markdown("""
        Acum că modelul nostru este antrenat și evaluat, putem folosi acest model pentru a face predicții
        individuale pentru calitatea somnului. Selectați un index din setul de testare pentru a vedea
        cum modelul prezice calitatea somnului pentru acel caz specific și cât de aproape este față de valoarea reală.
        """)

        selected_index = st.selectbox("Selectați un index:", st.session_state.X_test.index)

        if selected_index is not None:
            model = st.session_state.lr_model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            apply_log = st.session_state.apply_log

            single_prediction = model.predict(X_test.loc[selected_index].values.reshape(1, -1))[0]
            actual_value = y_test[X_test.index.get_loc(selected_index)]

            if apply_log:
                single_prediction_original = np.exp(single_prediction) - 1
                actual_value_original = np.exp(actual_value) - 1
                st.markdown(f"**Valoare prezisă (original):** {single_prediction_original:.4f}")
                st.markdown(f"**Valoare reală (original):** {actual_value_original:.4f}")
            else:
                st.markdown(f"**Valoare prezisă:** {single_prediction:.4f}")
                st.markdown(f"**Valoare reală:** {actual_value:.4f}")

            st.markdown("**Detaliile cazului:**")
            st.dataframe(X_test.loc[[selected_index]])

            st.markdown("#### Analiză detaliată a predicției:")
            error = abs(single_prediction - actual_value)
            error_percent = (error / abs(actual_value)) * 100 if actual_value != 0 else float('inf')

            if apply_log:
                error_original = abs(single_prediction_original - actual_value_original)
                error_percent_original = (
                                                 error_original / abs(
                                             actual_value_original)) * 100 if actual_value_original != 0 else float(
                    'inf')
                col1, col2 = st.columns(2)
                col1.metric("Eroare absolută", f"{error_original:.4f}")
                col2.metric("Eroare procentuală", f"{error_percent_original:.2f}%")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Eroare absolută", f"{error:.4f}")
                col2.metric("Eroare procentuală", f"{error_percent:.2f}%")

            if error_percent < 5:
                st.success("🎯 Predicție excelentă! Eroarea este sub 5% din valoarea reală.")
            elif error_percent < 10:
                st.success("✅ Predicție bună. Eroarea este între 5-10% din valoarea reală.")
            elif error_percent < 20:
                st.warning("⚠️ Predicție acceptabilă. Eroarea este între 10-20% din valoarea reală.")
            else:
                st.error("❌ Predicție slabă. Eroarea depășește 20% din valoarea reală.")

    # Adăugăm o secțiune de interpretare generală a modelului
    st.subheader("Interpretarea generală a modelului")
    with st.expander("Interpretarea generală a rezultatelor"):
        st.markdown("""
        ### Cum să interpretezi rezultatele modelului de regresie:

        #### 1. Calitatea generală a modelului
        - **R² între 0.7-1.0**: Modelul este bun sau excelent și explică mare parte din variația datelor.
        - **R² între 0.5-0.7**: Modelul este acceptabil, dar lasă o parte semnificativă din variație neexplicată.
        - **R² sub 0.5**: Modelul este slab și ar trebui îmbunătățit.

        #### 2. Problemele comune și soluțiile lor

        **Dacă modelul are performanță slabă (R² scăzut):**
        - Factori importanți ar putea lipsi din model
        - Relația poate fi nelineară și ar necesita transformări ale variabilelor
        - Ar putea fi nevoie de un model mai complex (ex. Random Forest, Gradient Boosting)

        **Dacă modelul prezintă overfitting (performanță mult mai bună pe setul de antrenare):**
        - Reduceți numărul de caracteristici
        - Adăugați regularizare (Ridge sau Lasso regression)
        - Colectați mai multe date

        **Dacă predicțiile sunt sistematic deplasate într-o direcție:**
        - Verificați dacă lipsesc factori importanți
        - Verificați dacă datele sunt reprezentative pentru populația țintă

        #### 3. Utilizarea modelului pentru îmbunătățirea sănătății somnului

        Modelul poate ajuta la:
        - Identificarea factorilor cu cel mai mare impact asupra calității somnului
        - Prioritizarea schimbărilor de stil de viață pentru îmbunătățirea somnului
        - Crearea de recomandări personalizate bazate pe caracteristicile individuale

        """)

    return data