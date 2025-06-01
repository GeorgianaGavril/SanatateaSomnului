import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

    # 1. Selectarea variabilelor pentru clustering
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

    # Opțional: variabile suplimentare
    additional_vars = st.multiselect(
        "Selectați variabile suplimentare pentru clustering (opțional):",
        [col for col in numerical_columns if col not in [var1, var2]],
        default=[]
    )

    # Combina toate variabilele selectate
    selected_vars = [var1, var2] + additional_vars

    # 2. Pregătirea datelor pentru clustering
    st.subheader("2. Pregătirea datelor pentru clustering")

    st.markdown("""
    Datele au fost deja scalate înainte de a ajunge aici, deci nu mai aplicăm nicio scalare suplimentară.
    """)

    # Extragem datele selectate (presupunem că sunt deja scalate upstream)
    X_raw = data[selected_vars].values

    # Folosim direct X_raw mai departe
    X = X_raw

    st.success(f"Datele pregătite pentru clustering. Forma datelor: {X.shape}")

    # 3. Determinarea numărului optim de clustere
    st.subheader("3. Determinarea numărului optim de clustere")

    st.markdown("""
    Pentru a determina numărul optim de clustere, vom folosi două metode:

    1. **Metoda Elbow (Cotului)**: plotăm WCSS (Within-Cluster Sum of Squares) pentru diferite k și căutăm "cotul".  
    2. **Scorul Silhouette**: măsoară cât de bine sunt aglomerate punctele în interiorul fiecărui cluster.
    """)

    wcss = []
    silhouette_scores = []
    k_range = range(2, min(11, len(data) - 1))  # Maxim k = 10 sau len(data)-1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, k in enumerate(k_range):
        status_text.text(f"Calculez pentru k = {k}...")
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(X, labels))

        progress_bar.progress((i + 1) / len(k_range))

    status_text.text("Calcule finalizate!")

    # Grafic WCSS (Elbow)
    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(list(k_range), wcss, marker='o', linestyle='-', color='red')
    ax_elbow.set_xlabel('Număr de clustere')
    ax_elbow.set_ylabel('WCSS')
    ax_elbow.set_title('Metoda Elbow pentru determinarea numărului de clustere')
    ax_elbow.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_elbow)

    st.markdown("""
    **Interpretare Elbow:** Căutați „cotul” în grafic – punctul în care WCSS nu mai scade semnificativ.
    """)

    # Grafic Silhouette
    fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
    ax_sil.plot(list(k_range), silhouette_scores, marker='o', linestyle='-', color='blue')
    ax_sil.set_xlabel('Număr de clustere')
    ax_sil.set_ylabel('Scor Silhouette')
    ax_sil.set_title('Scorul Silhouette pentru diferite numere de clustere')
    ax_sil.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_sil)

    st.markdown("""
    **Interpretare Silhouette:**  
    - Aproape de 1 → clustere bine distincte  
    - Aproape de 0 → puncte la granița clusterelor  
    - < 0 → posibil atribuire greșită  
    """)

    # Rezultate tabelare
    results_df = pd.DataFrame({
        'Număr clustere': list(k_range),
        'WCSS': wcss,
        'Silhouette': silhouette_scores
    })
    st.dataframe(results_df)

    optimal_k = k_range[np.argmax(silhouette_scores)]
    st.info(f"Conform Silhouette, numărul optim de clustere este: **{optimal_k}**")

    # 4. Aplicarea K-means cu numărul ales
    st.subheader("4. Aplicarea algoritmului K-means")

    n_clusters = st.slider(
        "Selectați numărul de clustere:",
        min_value=2,
        max_value=min(10, len(data) - 1),
        value=optimal_k
    )

    kmeans_final = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(X)

    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels

    final_silhouette = silhouette_score(X, cluster_labels)

    st.success(f"""
    Clustering final:  
    - k = {n_clusters}  
    - Silhouette Score final: {final_silhouette:.4f}
    """)

    # 5. Vizualizarea clusterelor în 2D
    st.subheader("5. Vizualizarea clusterelor")

    # Extragem primele două dimensiuni (var1 și var2) din X_raw
    X_2d = X_raw[:, :2]  # doar primele 2 caracteristici selectate

    viz_df = pd.DataFrame({
        var1: X_2d[:, 0],
        var2: X_2d[:, 1],
        'Cluster': cluster_labels
    })

    # Centroizii obținuți direct (deoarece nu există scalare internă)
    centroids = kmeans_final.cluster_centers_

    centroids_df = pd.DataFrame({
        var1: centroids[:, 0],
        var2: centroids[:, 1]
    })

    fig_clusters, ax_clusters = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("tab10", n_clusters)

    # Scatter plot cu clustere
    sns.scatterplot(
        data=viz_df,
        x=var1,
        y=var2,
        hue='Cluster',
        palette=palette,
        s=80,
        alpha=0.7,
        ax=ax_clusters
    )

    # Adăugăm centroizii (în aceeași scală!)
    sns.scatterplot(
        data=centroids_df,
        x=var1,
        y=var2,
        s=200,
        color='black',
        marker='X',
        edgecolor='white',
        linewidth=1.5,
        label='Centroizi',
        ax=ax_clusters
    )

    ax_clusters.set_title(f"Clustere K-means (k={n_clusters})", fontsize=14)
    ax_clusters.set_xlabel(var1, fontsize=12)
    ax_clusters.set_ylabel(var2, fontsize=12)
    ax_clusters.grid(True, linestyle='--', alpha=0.3)
    ax_clusters.legend(title='Cluster')

    st.pyplot(fig_clusters)

    # 6. Profilul clusterelor
    st.subheader("6. Profilul clusterelor")

    cluster_profiles = data_with_clusters.groupby('Cluster')[selected_vars].mean()

    st.write("**Mediile variabilelor pentru fiecare cluster:**")
    st.dataframe(cluster_profiles)

    fig_profiles, ax_profiles = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cluster_profiles,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
        linewidths=.5,
        ax=ax_profiles
    )
    ax_profiles.set_title("Profilul clusterelor (valori medii)", fontsize=14)
    st.pyplot(fig_profiles)

    # 7. Interpretarea clusterelor
    st.subheader("7. Interpretarea clusterelor")

    for i in range(n_clusters):
        st.markdown(f"##### Cluster {i}:")
        cluster_profile = cluster_profiles.loc[i]
        global_means = data[selected_vars].mean()
        diff_pct = ((cluster_profile - global_means) / global_means * 100).round(1)
        sorted_feats = diff_pct.abs().sort_values(ascending=False)
        top_feats = sorted_feats.index[: min(5, len(sorted_feats))]

        for feat in top_feats:
            val = cluster_profile[feat]
            diff = diff_pct[feat]
            direction = "mai mare" if diff > 0 else "mai mic"
            st.markdown(f"- **{feat}**: {val:.2f} ({abs(diff):.1f}% {direction} decât media)")

        size = (data_with_clusters['Cluster'] == i).sum()
        pct = (size / len(data_with_clusters) * 100).round(1)
        st.markdown(f"- **Dimensiune**: {size} indivizi ({pct}% din total)")
        st.markdown("---")

    # 8. Export date cu etichete de cluster
    st.subheader("8. Export date cu clustere")

    if st.button("Descarcă CSV cu etichete de clustere"):
        csv = data_with_clusters.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descarcă",
            data=csv,
            file_name="date_cu_clustere.csv",
            mime="text/csv"
        )

    # 9. Concluzii și aplicații
    st.subheader("9. Concluzii și aplicații")

    st.markdown("""
    - Segmentarea participanților în grupuri omogene poate ajuta la personalizarea recomandărilor de îmbunătățire a somnului.  
    - Tipologii diferite de utilizatori pot avea nevoi și riscuri diferite (ex.: clustere cu somn scăzut și stres ridicat).  
    - Ulterior, fiecare cluster poate fi folosit ca segment țintă pentru modele predictive (ex.: regresie liniară pe fiecare cluster)  
      sau intervenții specializate.
    """)

    if st.checkbox("Adaugă etichetele de cluster la setul de date"):
        return data_with_clusters
    else:
        return data
