import streamlit as st
# Funcții pentru fiecare pagină
def page_prezentare_generala(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Sănătatea somnului</h1>',
                unsafe_allow_html=True)

    st.markdown(r"""
        Acest set de date este preluat de pe Kaggle și conține informații despre sănătatea somnului, inclusiv durata somnului, calitatea somnului, nivelul de activitate fizică și stres, plus factori de sănătate (BMI, tensiune arterială, ritm cardiac, pași zilnici, tulburări de somn).
        Are 374 de înregistrări și 13 coloane.
        """)
    st.write(data)

    st.subheader("Variabilele setului de date:\n")
    st.markdown(r"""
                - **Person ID:** Reprezintă ID-ul fiecărei persoane care a participat la studiu.
                - **Gender:** Genul fiecărei persoane - masculin sau feminin.
                - **Age:** Vârsta fiecărei persoane.
                - **Occupation:** Reprezintă profesia persoanei respective.
                - **Sleep Duration:** Indică numărul de ore de somn pe care le are persoana în fiecare zi.
                - **Quality of Sleep:** Pe o scara de la 1 la 10 - constituie o evaluare subiectivă a calității somnului.
                - **Physical Activity Level:** Timpul petrecut efectuând activități fizice zilnic măsurat în minute.
                - **Stress Level:**  Reprezintă o evaluare subiectivă a nivelului de stres al persoanei, măsurat pe o scală de la 1 la 10.
                - **BMI Category:** Constituie categoria în care se încadrează persoana respectivă în funcție de IMC (indicele de masă corporală).
                - **Blood Pressure:** Tensiunea arterială a persoanei, indicată ca raportul dintre presiunea sistolică și presiunea diastolică.
                - **Heart Rate:** Ritmul cardiac de repaus al persoanei în bătăi pe minut.
                - **Daily Steps:** Numărul de pași pe care îl face persoana respectivă.
                - **Sleep Disorder:** Indică tulburarea de somn (dacă există) a persoanei respective.""")
    return data