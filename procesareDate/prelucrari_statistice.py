import streamlit as st
import matplotlib.pyplot as plt


def page_prelucrari_statistice(data):
    st.title("Prelucrări statistice și agregări")

    st.write("### Alege o variabilă pentru grupare:")
    group_col = st.selectbox("Coloană de grupare", options=data.select_dtypes(include=['object']).columns)

    st.write("### Alege o variabilă numerică pentru calcul:")
    num_col = st.selectbox("Coloană numerică", options=data.select_dtypes(include=['int64', 'float64']).columns)

    if group_col and num_col:
        st.write(f"#### Statistici pentru `{num_col}` grupate după `{group_col}`:")
        grouped_stats = data.groupby(group_col)[num_col].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
        st.dataframe(grouped_stats)

        # Grafic bară cu media
        st.write(f"### Grafic: media valorii `{num_col}` pe categorii de `{group_col}`")
        fig, ax = plt.subplots()
        grouped_stats.plot(kind='bar', x=group_col, y='mean', legend=False, ax=ax)
        ax.set_ylabel("Media")
        st.pyplot(fig)

    return data
