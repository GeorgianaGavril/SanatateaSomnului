import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def page_analiza_corelatiilor(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Analiza corelațiilor între variabile</h1>',
                unsafe_allow_html=True)

    st.markdown("Vom identifica relațiile liniare între variabilele numerice prin intermediul matricei de corelații.")

    numerical_cols = data.select_dtypes(include=[np.number]).columns

    corr_matrix = data[numerical_cols].corr()

    # Vizualizăm matricea de corelație cu un heatmap
    plt.figure(figsize=(5, 4))
    sns.set(font_scale=0.5)
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title("Matricea de corelație pentru variabilele numerice", fontsize=9)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.markdown(r"""
        **Interpretări**
        - Între Sleep Duration și Quality of Sleep există o corelație pozitivă puternică, de 0.88.
         Cu cât o persoană doarme mai mult, cu atât este mai probabil ca somnul să fie de o calitate mai bună.
        - Physical Activity Level are, de asemenea, o corelație pozitivă puternică cu Daily Steps, de 0.77. 
        Asta se întâmplă pentru că un număr mai mare de pași zilnici corespunde unui nivel de activitate fizică mai mare.
        - Și Stress Level are o corelație pozitivă semnificativă, de 0.67, cu Heart Rate, ceea ce sugerează că un nivel 
        crescut de stres determină o accelerare a ritmului cardiac.
    """)
    st.markdown(r"""
        - Quality of Sleep și Stress Level au o corelație negativă puternică, de -0.90, întrucât persoanele 
        care au un nivel de stres ridicat nu se mai odihnesc la fel de bine.
        - Stress Level are un impact negativ puternic și asupra lui Sleep Duration (-0.81), semnalând că persoanele stresate tind să doarmă mai puțin.
        - Între Quality of Sleep și Heart Rate există corelație negativă de -0.66, adică ritmul inimii accelerat poate afecta somnul.
    """)

    st.markdown("**Concluzie**")
    st.markdown("În ansamblu, se observă că stresul are un impact negativ asupra somnului, iar activitatea fizică susține "
                "un stil de viață mai echilibrat. Corelațiile identificate pot sta la baza unor analize predictive sau a unor "
                "recomandări personalizate privind sănătatea somnului.")

    return data