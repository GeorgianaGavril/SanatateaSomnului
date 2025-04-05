import streamlit as st
import math
import matplotlib.pyplot as plt
import numpy as np

def page_analiza_distributie(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Analiza distribuției datelor</h1>',
                unsafe_allow_html=True)

    st.markdown(
        "Vom analiza cum sunt distribuite datele prin intermediul unor histograme. Acestea împart valorile variabilelor numerice în intervale, evidențiind frecvența fiecărui interval.")

    # Verificăm coloanele numerice
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # Generarea histogramelor pentru variabilele numerice
    n_cols = 3  # Stabilim că dorim 3 grafice pe rând
    n_rows = math.ceil(
        len(numeric_columns) / n_cols)  # Calculăm numărul de rânduri necesare în funcție de numărul total de coloane numerice
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Setăm dimensiunea totală a figurii (lățime și înălțime)

    # Iterăm prin fiecare coloană numerică și generăm histogramă
    for i, col in enumerate(numeric_columns):
        plt.subplot(n_rows, n_cols,
                    i + 1)  # Creăm un subplot în grila de n_rows x n_cols; i+1 pentru indexarea subgraficelor începând de la 1
        plt.hist(data[col].dropna(), bins=30, edgecolor='black',
                 color='skyblue')  # Construim histograma pentru coloana curentă, eliminând valorile lipsă
        plt.title(f'Distribuția: {col}')  # Setăm titlul graficului cu numele coloanei
        plt.xlabel(col)  # Etichetă pentru axa x, indicând numele variabilei
        plt.ylabel('Frecvență')  # Etichetă pentru axa y, indicând frecvența valorilor
    plt.tight_layout()  # Ajustăm automat spațiile dintre subgrafice pentru a evita suprapunerea
    st.pyplot(plt)

    st.markdown("""
    ## Interpretarea distribuțiilor:

    - **Age (Vârstă)**: Distribuție multimodală cu un vârf principal la aproximativ 45 de ani. Majoritatea subiecților sunt adulți de vârstă medie.

    - **Sleep Duration (Durata Somnului)**: Distribuție relativ uniformă între 6 și 8.5 ore, cu vârfuri la 6.5, 7 și 7.5 ore, ceea ce indică pattern-uri tipice de somn.

    - **Quality of Sleep (Calitatea Somnului)**: Majoritatea subiecților raportează calitate bună a somnului (6-8 pe scară), cu foarte puțini raportând calitate scăzută.

    - **Physical Activity Level (Nivelul de Activitate Fizică)**: Distribuție multimodală cu concentrări la valori "rotunde", sugerând posibil o tendință de raportare aproximativă.

    - **Stress Level (Nivelul de Stres)**: Distribuție uniformă între 3 și 8, indicând o varietate de niveluri de stres în eșantion.

    - **Heart Rate (Rata Cardiacă)**: Prezintă asimetrie pozitivă, cu majoritatea valorilor între 65-75 bpm, dar cu o coadă spre dreapta indicând posibile outlier-uri care necesită investigare.

    - **Daily Steps (Pași Zilnici)**: Vârf principal la aproximativ 8000 pași, cu majoritate subiecților raportând între 5000-8000 pași zilnici.
    """)

    return data