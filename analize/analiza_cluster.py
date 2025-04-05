import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import streamlit as st


def page_analiza_clustering(data):
    st.title("Analiza de clustering K-means")

    st.markdown("""
    În această secțiune, vom aplica algoritmul K-means pentru a identifica grupuri (clustere) 
    în setul de date privind sănătatea somnului. Clusteringul poate ajuta la identificarea 
    tipologiilor de indivizi cu caracteristici similare privind somnul și stilul de viață.
    """)

    if not isinstance(data, pd.DataFrame):
        st.error("Nu există date disponibile pentru analiză.")
        return data

    # Verificăm dacă avem suficiente date
    if len(data) < 10:
        st.error("Nu există suficiente date pentru analiza de clustering.")
        return data

    # Selectarea variabilelor pentru clustering
    st.subheader("1. Selectarea variabilelor pentru clustering")

    # Obținem doar coloanele numerice
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    st.markdown("""
    Pentru clustering, vom folosi doar variabile numerice. Selectați două variabile pentru 
    vizualizarea clusterelor (pentru vizualizare 2D). Ulterior, puteți selecta variabile 
    suplimentare pentru un clustering mai complex.
    """)

    # Selectarea a două variabile principale pentru vizualizare
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox(
            "Selectați prima variabilă:",
            numerical_columns,
            index=numerical_columns.index('Sleep Duration') if 'Sleep Duration' in numerical_columns else 0
        )

    with col2:
        remaining_columns = [col for col in numerical_columns if col != var1]
        var2 = st.selectbox(
            "Selectați a doua variabilă:",
            remaining_columns,
            index=remaining_columns.index('Quality of Sleep') if 'Quality of Sleep' in remaining_columns else 0
        )

    # Opțional, permite utilizatorului să aleagă variabile suplimentare
    additional_vars = st.multiselect(
        "Selectați variabile suplimentare pentru clustering (opțional):",
        [col for col in numerical_columns if col not in [var1, var2]],
        default=[]
    )

    # Combinăm toate variabilele selectate
    selected_vars = [var1, var2] + additional_vars

    # Pregătirea datelor pentru clustering
    st.subheader("2. Pregătirea datelor pentru clustering")

    st.markdown("""
    Înainte de a aplica algoritmul K-means, datele vor fi scalate pentru a asigura 
    că toate variabilele contribuie în mod egal la analiză, indiferent de unitățile lor de măsură.
    """)

    # Extragem datele selectate
    X_raw = data[selected_vars].values

    # Scalarea datelor
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    st.success(f"Datele au fost scalate cu succes. Forma datelor: {X.shape}")

    # Determinarea numărului optim de clustere
    st.subheader("3. Determinarea numărului optim de clustere")

    st.markdown("""
    Pentru a determina numărul optim de clustere, vom folosi două metode:

    1. **Metoda Elbow (Cotului)**: Plotăm WCSS (Within-Cluster Sum of Squares) pentru 
       diferite valori ale k și căutăm "cotul" în grafic.
    2. **Scorul Silhouette**: Măsoară cât de similare sunt obiectele în propriul cluster 
       comparativ cu alte clustere.
    """)

    # Calculăm WCSS pentru diferite valori ale k
    wcss = []
    silhouette_scores = []
    k_range = range(2, min(11, len(data) - 1))  # Limitează k la n-1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, k in enumerate(k_range):
        status_text.text(f"Calculez pentru k = {k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

        # Calculăm scorul silhouette pentru k ≥ 2
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))

        # Actualizăm bara de progres
        progress_bar.progress((i + 1) / len(k_range))

    status_text.text("Calcule finalizate!")

    # Afișăm graficul WCSS (Metoda Elbow)
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 6))
    ax_elbow.plot(list(k_range), wcss, marker='o', linestyle='-', color='red')
    ax_elbow.set_xlabel('Număr de clustere')
    ax_elbow.set_ylabel('WCSS')
    ax_elbow.set_title('Metoda Elbow pentru determinarea numărului optim de clustere')
    ax_elbow.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_elbow)

    st.markdown("""
    **Interpretarea metodei Elbow**: 
    Căutăm "cotul" în grafic, punctul unde adăugarea de clustere suplimentare nu reduce 
    semnificativ WCSS. Acesta este considerat numărul optim de clustere.
    """)

    # Afișăm graficul Silhouette
    fig_silhouette, ax_silhouette = plt.subplots(figsize=(10, 6))
    ax_silhouette.plot(list(k_range), silhouette_scores, marker='o', linestyle='-', color='blue')
    ax_silhouette.set_xlabel('Număr de clustere')
    ax_silhouette.set_ylabel('Scor Silhouette')
    ax_silhouette.set_title('Scorul Silhouette pentru diferite numere de clustere')
    ax_silhouette.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig_silhouette)

    st.markdown("""
    **Interpretarea scorului Silhouette**:
    - ~1: Punctele sunt bine atribuite clusterelor lor
    - ~0: Punctele sunt la granița între clustere
    - < 0: Punctele sunt probabil atribuite greșit

    Căutăm valoarea k care maximizează scorul Silhouette.
    """)

    # Afișăm rezultatele numerice
    results_df = pd.DataFrame({
        'Număr clustere': list(k_range),
        'WCSS': wcss,
        'Scor Silhouette': silhouette_scores
    })
    st.dataframe(results_df)

    # Recomandări pentru numărul optim de clustere
    optimal_k_silhouette = k_range[silhouette_scores.index(max(silhouette_scores))]

    st.info(f"""
    **Recomandare**: Conform scorului Silhouette, numărul optim de clustere este: **{optimal_k_silhouette}**

    Notă: Metoda Elbow necesită o interpretare vizuală, căutați "cotul" în graficul WCSS.
    """)

    # Alegerea numărului de clustere
    st.subheader("4. Aplicarea algoritmului K-means")

    n_clusters = st.slider(
        "Selectați numărul de clustere:",
        min_value=2,
        max_value=min(10, len(data) - 1),
        value=optimal_k_silhouette
    )

    # Aplicarea K-means cu numărul ales de clustere
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # Adăugăm etichetele de cluster la dataframe
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels

    # Calculăm scorul silhouette pentru validare
    final_silhouette = silhouette_score(X, cluster_labels)

    st.success(f"""
    Clusteringul a fost realizat cu succes!

    - Număr de clustere: {n_clusters}
    - Scorul Silhouette final: {final_silhouette:.4f}
    """)

    # Vizualizarea clusterelor în 2D
    st.subheader("5. Vizualizarea clusterelor")

    # Extragem datele pentru primele două variabile pentru vizualizare
    X_2d = X_raw[:, 0:2]  # Luăm doar primele două variabile

    # Creem un DataFrame pentru vizualizare
    viz_df = pd.DataFrame({
        var1: X_2d[:, 0],
        var2: X_2d[:, 1],
        'Cluster': cluster_labels
    })

    # Convertim centroids înapoi la scala originală
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Creem un DataFrame pentru centroizi
    centroids_df = pd.DataFrame({
        var1: centroids_original[:, 0],
        var2: centroids_original[:, 1],
    })

    # Afișăm graficul de dispersie cu clustere
    fig_clusters, ax_clusters = plt.subplots(figsize=(12, 8))

    # Definim o paletă de culori pentru clustere
    palette = sns.color_palette("hls", n_clusters)

    # Creăm scatter plot-ul cu clustere
    scatter = sns.scatterplot(
        data=viz_df,
        x=var1,
        y=var2,
        hue='Cluster',
        palette=palette,
        s=100,
        alpha=0.7,
        ax=ax_clusters
    )

    # Adăugăm centroizii
    sns.scatterplot(
        data=centroids_df,
        x=var1,
        y=var2,
        s=200,
        color='red',
        marker='X',
        edgecolor='black',
        linewidth=1,
        label='Centroizi',
        ax=ax_clusters
    )

    # Stilizăm graficul
    ax_clusters.set_title(f'Clustere K-means (k={n_clusters})', fontsize=16)
    ax_clusters.set_xlabel(var1, fontsize=14)
    ax_clusters.set_ylabel(var2, fontsize=14)
    ax_clusters.grid(True, linestyle='--', alpha=0.3)

    # Afișăm legenda
    ax_clusters.legend(title='Cluster', title_fontsize=12)

    st.pyplot(fig_clusters)

    # Analizăm caracteristicile fiecărui cluster
    st.subheader("6. Profilul clusterelor")

    st.markdown("""
    Să analizăm caracteristicile fiecărui cluster pentru a înțelege ce tipuri de grupuri au fost identificate.
    """)

    # Calculăm mediile pentru fiecare variabilă în fiecare cluster
    cluster_profiles = data_with_clusters.groupby('Cluster')[selected_vars].mean()

    # Afișăm profilurile clusterelor
    st.write("**Media variabilelor pentru fiecare cluster:**")
    st.dataframe(cluster_profiles)

    # Vizualizăm profilurile clusterelor
    fig_profiles = plt.figure(figsize=(14, 8))
    ax_profiles = sns.heatmap(
        cluster_profiles,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=.5
    )
    plt.title('Profilul clusterelor (valori medii)', fontsize=16)
    st.pyplot(fig_profiles)

    # Interpretarea clusterelor
    st.subheader("7. Interpretarea clusterelor")

    st.markdown("""
    În funcție de valorile medii ale variabilelor în fiecare cluster, putem interpreta și caracteriza fiecare grup:
    """)

    for i in range(n_clusters):
        st.markdown(f"##### Cluster {i}:")

        # Selectăm cele mai distinctive caracteristici pentru acest cluster
        cluster_profile = cluster_profiles.loc[i]
        global_means = data[selected_vars].mean()

        # Calculăm diferența procentuală față de media globală
        diff_pct = ((cluster_profile - global_means) / global_means * 100).round(1)

        # Sortăm caracteristicile după valoarea absolută a diferenței
        sorted_features = diff_pct.abs().sort_values(ascending=False)

        # Luăm primele 3-5 caracteristici (sau toate dacă sunt mai puține)
        top_features = min(5, len(sorted_features))
        distinctive_features = sorted_features.index[:top_features]

        # Afișăm caracteristicile distinctive
        for feature in distinctive_features:
            value = cluster_profile[feature]
            diff = diff_pct[feature]
            direction = "mai mare" if diff > 0 else "mai mic"

            st.markdown(f"- **{feature}**: {value:.2f} ({abs(diff):.1f}% {direction} decât media)")

        # Calculăm mărimea clusterului
        cluster_size = (data_with_clusters['Cluster'] == i).sum()
        cluster_pct = (cluster_size / len(data_with_clusters) * 100).round(1)

        st.markdown(f"- **Mărime**: {cluster_size} indivizi ({cluster_pct}% din total)")
        st.markdown("---")

    # Oferim opțiunea de a descărca datele cu etichetele de cluster
    st.subheader("8. Export date cu clustere")

    if st.button("Pregătește datele cu etichete de cluster pentru export"):
        csv = data_with_clusters.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descarcă CSV cu clustere",
            data=csv,
            file_name="date_cu_clustere.csv",
            mime="text/csv"
        )

    # Concluzii și recomandări
    st.subheader("9. Concluzii și aplicații")

    st.markdown("""
    ### Utilitatea analizei de clustering:

    1. **Segmentarea populației**: Identificarea unor tipologii distincte de persoane 
       în funcție de caracteristicile somnului și stilului de viață.

    2. **Intervenții personalizate**: Dezvoltarea unor strategii specifice pentru 
       îmbunătățirea somnului pentru fiecare segment.

    3. **Analiză aprofundată**: Clusterele pot fi studiate în detaliu pentru a înțelege 
       mai bine relațiile dintre variabilele care caracterizează somnul și stilul de viață.

    4. **Predicții**: Modelele predictive (precum regresia liniară) pot fi aplicate separat 
       pentru fiecare cluster, potențial îmbunătățind acuratețea predicțiilor.
    """)

    # Returnăm datele originale sau cele cu etichete de cluster în funcție de opțiunea utilizatorului
    if st.checkbox("Adaugă etichetele de cluster la setul de date"):
        return data_with_clusters
    else:
        return data