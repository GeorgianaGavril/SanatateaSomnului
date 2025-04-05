import streamlit as st

def page_tratare_valori_duplicate_nule(data):
    st.markdown(
        '<h1 style="color: #090909; font-size: 40px; text-align: center;">Tratarea valorilor duplicat și nule</h1>',
        unsafe_allow_html=True)

    st.markdown(
        "În continuare, vom elimina valorile duplicate și, fiind 242 astfel de valori, vom rămâne cu 132 de înregistrări.")

    cleaned_data = data.drop_duplicates()
    st.code("set_date = set_date.drop_duplicates()")
    st.write(cleaned_data)

    st.markdown("Verificăm dacă există valori nule:")
    st.code("set_date.isnull().sum()")
    st.write(cleaned_data.isnull().sum())

    st.markdown("""
    Obținem 73 de valori nule pentru coloana Sleep Disorder. Acest lucru se întâmplă din cauza faptului că acele persoane nu au nicio tulburare de somn, 
    deci vom înlocui aceste valori Nan cu textul „None”. Astfel, nu vor mai exista valori nule în setul de date.
    """)

    st.code("set_date['Sleep Disorder'] = set_date['Sleep Disorder'].fillna('None')")
    cleaned_data['Sleep Disorder'] = cleaned_data['Sleep Disorder'].fillna('None')
    st.write(cleaned_data)

    st.markdown(
        "Astfel, nu mai există valori nule în setul de date, așa că vom continua cu tratarea valorilor extreme.")

    return cleaned_data