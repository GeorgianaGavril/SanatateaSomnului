import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def page_scalare_date(data):
    st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Scalarea datelor numerice</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    Scalarea este procesul de aducere a datelor numerice la o scară comună. 
    """)

    # Make a deep copy of the original data to ensure we're not modifying it
    original_data = data.copy(deep=True)

    # Store the original data in session state if it's not already there
    if 'original_data_for_scaling' not in st.session_state:
        st.session_state.original_data_for_scaling = original_data

    # Always use the stored original data
    data_for_analysis = st.session_state.original_data_for_scaling

    # Identifică coloanele numerice pentru scalare
    numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude Person ID din scalare dacă există
    if 'Person ID' in numeric_cols:
        numeric_cols.remove('Person ID')

    st.write(f"Coloane numerice pentru scalare: {numeric_cols}")

    # Initialize the original boxplot once and cache it
    if 'original_boxplot' not in st.session_state:
        plt.close('all')  # Ensure clean state
        fig_orig, ax_orig = plt.subplots(figsize=(12, 6))

        # Sample consistently
        if len(data_for_analysis) > 100:
            sample_data = data_for_analysis.sample(100, random_state=42)
        else:
            sample_data = data_for_analysis

        sns.boxplot(data=sample_data[numeric_cols], ax=ax_orig)
        ax_orig.set_xticklabels(ax_orig.get_xticklabels(), rotation=90)
        ax_orig.set_title('Distribuția datelor înainte de scalare')

        # Save the figure to session state
        st.session_state.original_boxplot = fig_orig
        st.session_state.sample_indices = sample_data.index

    # Display the cached original boxplot
    st.markdown("### Distribuția datelor înainte de scalare")
    st.pyplot(st.session_state.original_boxplot)

    # Selectbox pentru alegerea metodei de scalare
    scaling_method = st.selectbox(
        'Alegeți metoda de scalare:',
        ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler')
    )

    # Each time a new scaling method is selected, we work with the original data again
    data_to_scale = data_for_analysis.copy()

    # Get the same sample indices as used for the original plot
    if len(data_for_analysis) > 100:
        sample_indices = st.session_state.sample_indices
    else:
        sample_indices = data_for_analysis.index

    # Apply the selected scaling method
    if scaling_method == 'StandardScaler':
        st.markdown("### StandardScaler (Z-score normalization)")
        st.markdown("""
        Această metodă standardizează datele astfel încât să aibă medie 0 și deviație standard 1.
        Formula: z = (x - μ) / σ
        Este potrivită pentru datele care urmează o distribuție aproximativ normală.
        """)

        std_scaler = StandardScaler()
        scaled_data = data_to_scale.copy()
        scaled_data[numeric_cols] = std_scaler.fit_transform(scaled_data[numeric_cols])

        st.write("Date după StandardScaler:")
        st.write(scaled_data)

        # Create a new figure for the scaled data
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))
        scaled_sample = scaled_data.loc[sample_indices]
        sns.boxplot(data=scaled_sample[numeric_cols], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Distribuția datelor după StandardScaler')
        st.pyplot(fig)
        plt.close(fig)

    elif scaling_method == 'MinMaxScaler':
        st.markdown("### MinMaxScaler")
        st.markdown("""
        Această metodă transformă datele în intervalul [0, 1].
        Formula: x_scaled = (x - min) / (max - min)
        Este utilă când nu avem o distribuție normală sau când intervalul [0, 1] este important.
        """)

        minmax_scaler = MinMaxScaler()
        scaled_data = data_to_scale.copy()
        scaled_data[numeric_cols] = minmax_scaler.fit_transform(scaled_data[numeric_cols])

        st.write("Date după MinMaxScaler:")
        st.write(scaled_data)

        # Create a new figure for the scaled data
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))
        scaled_sample = scaled_data.loc[sample_indices]
        sns.boxplot(data=scaled_sample[numeric_cols], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Distribuția datelor după MinMaxScaler')
        st.pyplot(fig)
        plt.close(fig)

    elif scaling_method == 'RobustScaler':
        st.markdown("### RobustScaler")
        st.markdown("""
        Această metodă utilizează IQR în loc de medie și deviație standard.
        Formula: x_scaled = (x - Q2) / (Q3 - Q1)
        Este robustă în prezența outlierilor și potrivită pentru date care conțin valori extreme.
        """)

        robust_scaler = RobustScaler()
        scaled_data = data_to_scale.copy()
        scaled_data[numeric_cols] = robust_scaler.fit_transform(scaled_data[numeric_cols])

        st.write("Date după RobustScaler:")
        st.write(scaled_data)

        # Create a new figure for the scaled data
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))
        scaled_sample = scaled_data.loc[sample_indices]
        sns.boxplot(data=scaled_sample[numeric_cols], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Distribuția datelor după RobustScaler')
        st.pyplot(fig)
        plt.close(fig)

    else:  # MaxAbsScaler
        st.markdown("### MaxAbsScaler")
        st.markdown("""
        Această metodă scalează fiecare caracteristică prin valoarea sa maximă absolută.
        Formula: x_scaled = x / max(|x|)
        Este utilă pentru date sparse și păstrează structura de date zero.
        """)

        maxabs_scaler = MaxAbsScaler()
        scaled_data = data_to_scale.copy()
        scaled_data[numeric_cols] = maxabs_scaler.fit_transform(scaled_data[numeric_cols])

        st.write("Date după MaxAbsScaler:")
        st.write(scaled_data)

        # Create a new figure for the scaled data
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))
        scaled_sample = scaled_data.loc[sample_indices]
        sns.boxplot(data=scaled_sample[numeric_cols], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_title('Distribuția datelor după MaxAbsScaler')
        st.pyplot(fig)
        plt.close(fig)

    # Comparație dintre metodele de scalare pentru o singură variabilă
    sample_feature = 'Stress Level'  # Alege o caracteristică pentru comparație

    # Creează un DataFrame pentru comparație
    comparison_df = pd.DataFrame({
        'Original': data_for_analysis[sample_feature]
    })

    # Adaugă datele scalate la DataFrame-ul de comparație
    comparison_df[scaling_method] = scaled_data[sample_feature]

    st.write(f"Comparație pentru variabila '{sample_feature}':")
    st.write(comparison_df.describe())

    # Create a completely new figure for comparison
    plt.close('all')
    fig_comp, ax_comp = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=comparison_df, ax=ax_comp)
    ax_comp.set_title(f'Comparație între datele originale și {scaling_method} pentru {sample_feature}')
    ax_comp.set_xticklabels(ax_comp.get_xticklabels(), rotation=45)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    # Salvarea datelor codificate și scalate
    st.markdown("### Salvarea datelor scalate")

    if st.button('Salvează datele procesate ca CSV'):
        filename = f'sleep_health_{scaling_method.lower()}.csv'
        scaled_data.to_csv(filename)
        st.success(f'Datele au fost salvate cu succes folosind {scaling_method}!')

    return scaled_data

