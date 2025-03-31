import math

import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

set_date = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", index_col=0)

st.markdown('<h1 style="color: #090909; font-size: 40px; text-align: center;">Sănătatea somnului</h1>', unsafe_allow_html=True)

st.markdown(r"""
    Acest set de date este preluat de pe Kaggle și conține informații despre sănătatea somnului, inclusiv durata somnului, calitatea somnului, nivelul de activitate fizică și stres, plus factori de sănătate (BMI, tensiune arterială, ritm cardiac, pași zilnici, tulburări de somn).
    Are 374 de înregistrări și 13 coloane.
    """)
st.write(set_date)

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

st.markdown("În continuare, vom elimina valorile duplicate și, fiind 242 astfel de valori, vom rămâne cu 132 de înregistrări.")

set_date = set_date.drop_duplicates()
st.code("set_date = set_date.drop_duplicates()")
st.write(set_date)

st.markdown("Verificăm dacă există valori nule:")
st.code("set_date.isnull().sum()")
print(set_date.isnull().sum())
st.write(set_date.isnull().sum())
st.markdown("Obținem 73 de valori nule pentru coloana Sleep Disorder. Acest lucru se întâmplă din cauza faptului că acele persoane nu au nicio tulburare de somn, "
            "deci vom înlocui aceste valori Nan cu textul „None”. Astfel, nu vor mai exista valori nule în setul de date.")

st.code("set_date['Sleep Disorder'] = set_date['Sleep Disorder'].fillna('None')")
set_date['Sleep Disorder'] = set_date['Sleep Disorder'].fillna('None')
st.write(set_date)

st.markdown("Astfel, nu mai există valori nule în setul de date, așa că vom continua cu tratarea valorilor extreme.")


#########################################################
# Analiza distributiei datelor
st.markdown("### Analiza distribuției datelor")

st.markdown("Vom analiza cum sunt distribuite datele prin intermediul unor histograme. Acestea împart valorile variabilelor numerice în intervale, evidențiind frecvența fiecărui interval.")

# Verificăm coloanele numerice
numeric_columns = set_date.select_dtypes(include=[np.number]).columns.tolist()

# Generarea histogramelor pentru variabilele numerice
n_cols = 3                                  # Stabilim că dorim 3 grafice pe rând
n_rows = math.ceil(len(numeric_columns) / n_cols)  # Calculăm numărul de rânduri necesare în funcție de numărul total de coloane numerice
plt.figure(figsize=(6 * n_cols, 4 * n_rows))  # Setăm dimensiunea totală a figurii (lățime și înălțime)

# Iterăm prin fiecare coloană numerică și generăm histogramă
for i, col in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)       # Creăm un subplot în grila de n_rows x n_cols; i+1 pentru indexarea subgraficelor începând de la 1
    plt.hist(set_date[col].dropna(), bins=30, edgecolor='black', color='skyblue')  # Construim histograma pentru coloana curentă, eliminând valorile lipsă
    plt.title(f'Distribuția: {col}')         # Setăm titlul graficului cu numele coloanei
    plt.xlabel(col)                          # Etichetă pentru axa x, indicând numele variabilei
    plt.ylabel('Frecvență')                    # Etichetă pentru axa y, indicând frecvența valorilor
plt.tight_layout()                          # Ajustăm automat spațiile dintre subgrafice pentru a evita suprapunerea
st.pyplot(plt)

st.markdown("""Dintre histogramele generate, cea pentru variabila Heart Rate prezintă o distribuție negativă, având o coadă lungă spre dreapta.
    Acest lucru poate indica prezența unor valori extreme, lucru pe care îl vom investiga mai departe prin metodele de tratare a outlierilor.""")


####################################################
# Tratarea valorilor extreme

st.header("Tratarea valorilor extreme")
# 1. Metoda IQR
st.subheader("1. Metoda IQR")

st.markdown(r"""
            Metoda Interquartile Range (IQR) este folosită pentru a detecta și elimina valorile extreme dintr-un set de date.
            - **Q1:** 25% din valorile cele mai mici
            - **Q3:** 75% din valorile cele mai mici
            - **IQR:** Q3 - Q1
            - **Outlieri:** valorile care sunt în afara intervalului acceptabil **[Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]**""")

# Calculate IQR and identify outliers for each numeric column

for column in numeric_columns:
    st.markdown(f"#### Analiza outlierilor pentru: {column}")

    Q1 = set_date[column].quantile(0.25)
    Q3 = set_date[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = set_date[(set_date[column] < lower_bound) | (set_date[column] > upper_bound)]

    st.write(f"Limita inferioară: {lower_bound}")
    st.write(f"Limita superioară: {upper_bound}")
    st.write(f"Număr de outlieri: {len(outliers)}")

    # Create boxplot to visualize outliers
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(y=set_date[column], ax=ax)
    ax.set_title(f'Boxplot pentru {column}')
    st.pyplot(fig)

    # Display the outliers if any exist
    if len(outliers) > 0:
        st.write("Valorile outlier:")
        st.write(outliers[[column, 'Gender', 'Age', 'Occupation', 'Sleep Disorder']])

st.markdown("Singura variabilă cu valori extreme este Heart Rate. În general, un ritm cardiac de repaus mai mic este considerat normal. "
            "Observăm că persoanele respective au tulburări de somn, lucru ce poate contribui la un ritm cardiac mai ridicat. În plus, aceste persoane "
            "au meserii stresante și solicitante, precum doctor, avocat și programator. Așadar, vom elimina aceste valori.")

# Remove all outliers from dataset
set_date_1 = set_date.copy()
st.subheader("Setul de date după eliminarea outlierilor:")

while True:
    initial_shape = set_date_1.shape[0]  # Stocăm numărul inițial de rânduri

    for column in numeric_columns:
        Q1 = set_date_1[column].quantile(0.25)
        Q3 = set_date_1[column].quantile(0.75)
        IQR = Q3 - Q1

        set_date_1 = set_date_1[~((set_date_1[column] < (Q1 - 1.5 * IQR)) | (set_date_1[column] > (Q3 + 1.5 * IQR)))]

    if set_date_1.shape[0] == initial_shape:
        break

st.write(f"Număr de înregistrări după eliminarea outlierilor: {len(set_date_1)}")
st.write(set_date_1)

plt.figure(figsize=(6 * n_cols, 4 * n_rows))
for i, col in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(set_date_1[col].dropna(), bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribuția (log): {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')

plt.tight_layout()
st.pyplot(plt)

##############################################################
# 2. Metoda Z-Score
set_date_2 = set_date.copy()
st.subheader("2. Metoda Z-Score")
st.markdown("Metoda Z-Score determină cât de departe este o valoare de media setului de date, în unități de deviație standard. Se bazează pe formula:")
st.latex(r"z = \frac{X - \mu}{\sigma}")
st.markdown(r"""
            - **X** este valoarea dată
            - **μ** este media setului de date
            - **σ** este deviația standard""")
st.markdown("Dacă |Z| > 3, se consideră outlier.")

# Calculăm Z-Score pentru fiecare coloană numerică
z_scores = np.abs(stats.zscore(set_date_2[numeric_columns]))

# Definim pragul pentru outlieri (ex: 3)
threshold = 3

# Selectăm doar rândurile unde toate valorile sunt sub prag
set_date_2 = set_date_2[(z_scores < threshold).all(axis=1)]

# Afișăm noul set de date
st.write(f"Număr de înregistrări după eliminarea outlierilor (Z-Score): {len(set_date_2)}")
st.write(set_date_2)

plt.figure(figsize=(6 * n_cols, 4 * n_rows))
for i, col in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(set_date_2[col].dropna(), bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribuția (log): {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')

plt.tight_layout()
st.pyplot(plt)

##########################################################
# Metoda 3. Transformarea logaritmica a datelor
st.subheader("3. Metoda transformării logaritmice a datelor")

st.markdown("Logartimarea reduce asimetria distribuției și efectul valorilor mari. Mai întâi trebuie să ne asigurăm că datele sunt strict pozitive, "
            "deoarece logaritmul nu este definit pentru valori negative sau zero.")
# Verificăm existența valorilor zero sau negative
numeric_columns_new = [col for col in numeric_columns if not (set_date[col] <= 0).any()]

for col in numeric_columns:
    if col not in numeric_columns_new:
        st.warning(f"Coloana {col} conține valori zero sau negative, deci nu poate fi logaritmată.")

# Aplicăm log(1 + x)
set_date_3 = set_date.copy()
for col in numeric_columns_new:
    set_date_3[col] = np.log1p(set_date_3[col])

st.write("Setul de date după aplicarea transformării logaritmice:")
st.write(set_date_3)

plt.figure(figsize=(6 * n_cols, 4 * n_rows))
for i, col in enumerate(numeric_columns):
    plt.subplot(n_rows, n_cols, i + 1)
    plt.hist(set_date_3[col].dropna(), bins=30, edgecolor='black', color='skyblue')
    plt.title(f'Distribuția: {col}')
    plt.xlabel(col)
    plt.ylabel('Frecvență')
plt.tight_layout()
st.pyplot(plt)

st.markdown("Observăm că cele 3 transformări nu aduc modificări majore asupra setului de date. "
            "Mai departe vom continua cu setul de date rezultat din metoda IQR.")

set_date = set_date_1


###########################################################################
# Data Encoding Section
st.subheader("Codificarea datelor categorice:")

# Create a copy of the dataset for encoding
encoded_data = set_date.copy()

# Identify categorical columns
categorical_columns = set_date.select_dtypes(include=['object']).columns.tolist()
st.write(f"Coloane categorice identificate: {categorical_columns}")

# 1. Label Encoding (pentru variabile ordinale sau binare)
st.markdown("### 1. Label Encoding")
st.markdown(
    "Această metodă înlocuiește fiecare categorie cu un număr întreg. Este potrivită pentru variabile ordinale sau binare.")

label_encoder = LabelEncoder()
le_data = encoded_data.copy()

for column in ['Gender', 'Sleep Disorder']:
    le_data[f'{column}_Label'] = label_encoder.fit_transform(le_data[column])
    # Afișează maparea pentru claritate
    mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    st.write(f"Mapare pentru {column}: {mapping}")

st.write("Date după Label Encoding:")
st.write(le_data)

# 2. One-Hot Encoding (pentru variabile nominale)
st.markdown("### 2. One-Hot Encoding")
st.markdown(
    "Această metodă creează coloane noi pentru fiecare categorie. Este potrivită pentru variabile nominale fără ordine specifică.")

ohe_data = encoded_data.copy()
for column in ['BMI Category', 'Occupation']:
    # Crearea dummies și excluderea primei coloane pentru a evita colinearitatea
    dummies = pd.get_dummies(ohe_data[column], prefix=column, drop_first=True)
    # Adăugarea dummy variabilelor la dataset
    ohe_data = pd.concat([ohe_data, dummies], axis=1)

st.write("Date după One-Hot Encoding pentru BMI Category și Occupation:")
st.write(ohe_data)

# 3. Encoding pentru Blood Pressure (separare în sistolic și diastolic)
st.markdown("### 3. Encoding special pentru Blood Pressure")
st.markdown("Separăm valorile tensiunii arteriale în valori sistolice și diastolice.")

if 'Blood Pressure' in encoded_data.columns:
    bp_data = encoded_data.copy()
    # Extrage valorile sistolice și diastolice din formatul "120/80"
    bp_data['Systolic'] = bp_data['Blood Pressure'].str.split('/').str[0].astype(int)
    bp_data['Diastolic'] = bp_data['Blood Pressure'].str.split('/').str[1].astype(int)

    st.write("Date după separarea Blood Pressure:")
    st.write(bp_data)

# 4. Combinarea metodelor pentru un dataset final pregătit pentru analiză
st.markdown("### 4. Dataset final codificat")
st.markdown("Combinăm cele mai potrivite metode de codificare pentru a crea un dataset final pregătit pentru analiză.")

final_data = encoded_data.copy()

# Label Encoding pentru variabile binare
for column in ['Gender', 'Sleep Disorder']:
    final_data[f'{column}_Encoded'] = label_encoder.fit_transform(final_data[column])

# One-Hot Encoding pentru variabile nominale cu mai multe categorii
for column in ['BMI Category', 'Occupation']:
    dummies = pd.get_dummies(final_data[column], prefix=column, drop_first=True)
    final_data = pd.concat([final_data, dummies], axis=1)

# Procesare Blood Pressure
if 'Blood Pressure' in final_data.columns:
    final_data['Systolic'] = final_data['Blood Pressure'].str.split('/').str[0].astype(int)
    final_data['Diastolic'] = final_data['Blood Pressure'].str.split('/').str[1].astype(int)

# Eliminăm coloanele originale care au fost codificate
final_data_clean = final_data.drop(['Gender', 'BMI Category', 'Occupation', 'Sleep Disorder', 'Blood Pressure'], axis=1)

st.write("Dataset final codificat (cu coloanele originale înlocuite):")
st.write(final_data_clean)

# Salvarea datelor pentru analiză ulterioară
st.markdown("### Salvarea datelor codificate")
st.markdown("Datele pot fi salvate pentru analiză ulterioară.")

if st.button('Salvează datele codificate ca CSV'):
    final_data_clean.to_csv('sleep_health_encoded.csv')
    st.success('Datele au fost salvate cu succes!')

# SECȚIUNEA NOUĂ: Scalarea datelor
st.subheader("Scalarea datelor numerice:")
st.markdown("""
Scalarea este procesul de aducere a datelor numerice la o scară comună. 
""")

# Identifică coloanele numerice pentru scalare
numeric_cols = final_data_clean.select_dtypes(include=[np.number]).columns.tolist()

# Exclude Person ID din scalare dacă există
if 'Person ID' in numeric_cols:
    numeric_cols.remove('Person ID')

st.write(f"Coloane numerice pentru scalare: {numeric_cols}")

# Comparație vizuală înainte de scalare
st.markdown("### Distribuția datelor înainte de scalare")

fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = final_data_clean[numeric_cols].sample(min(100, len(final_data_clean)))
sns.boxplot(data=data_to_plot, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# 1. StandardScaler (Z-score normalization)
st.markdown("### 1. StandardScaler (Z-score normalization)")
st.markdown("""
Această metodă standardizează datele astfel încât să aibă medie 0 și deviație standard 1.
Formula: z = (x - μ) / σ
Este potrivită pentru datele care urmează o distribuție aproximativ normală.
""")

std_scaler = StandardScaler()
std_scaled_data = final_data_clean.copy()
std_scaled_data[numeric_cols] = std_scaler.fit_transform(std_scaled_data[numeric_cols])

st.write("Date după StandardScaler:")
st.write(std_scaled_data)

# Vizualizare date standardizate
fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = std_scaled_data[numeric_cols].sample(min(100, len(std_scaled_data)))
sns.boxplot(data=data_to_plot, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribuția datelor după StandardScaler')
st.pyplot(fig)

# 2. MinMaxScaler (normalizare în intervalul [0,1])
st.markdown("### 2. MinMaxScaler")
st.markdown("""
Această metodă transformă datele în intervalul [0, 1].
Formula: x_scaled = (x - min) / (max - min)
Este utilă când nu avem o distribuție normală sau când intervalul [0, 1] este important.
""")

minmax_scaler = MinMaxScaler()
minmax_scaled_data = final_data_clean.copy()
minmax_scaled_data[numeric_cols] = minmax_scaler.fit_transform(minmax_scaled_data[numeric_cols])

st.write("Date după MinMaxScaler:")
st.write(minmax_scaled_data)

# Vizualizare date normalizate MinMax
fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = minmax_scaled_data[numeric_cols].sample(min(100, len(minmax_scaled_data)))
sns.boxplot(data=data_to_plot, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribuția datelor după MinMaxScaler')
st.pyplot(fig)

# 3. RobustScaler (robust to outliers)
st.markdown("### 3. RobustScaler")
st.markdown("""
Această metodă utilizează IQR în loc de medie și deviație standard.
Formula: x_scaled = (x - Q2) / (Q3 - Q1)
Este robustă în prezența outlierilor și potrivită pentru date care conțin valori extreme.
""")

robust_scaler = RobustScaler()
robust_scaled_data = final_data_clean.copy()
robust_scaled_data[numeric_cols] = robust_scaler.fit_transform(robust_scaled_data[numeric_cols])

st.write("Date după RobustScaler:")
st.write(robust_scaled_data)

# Vizualizare date scalate robuste
fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = robust_scaled_data[numeric_cols].sample(min(100, len(robust_scaled_data)))
sns.boxplot(data=data_to_plot, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribuția datelor după RobustScaler')
st.pyplot(fig)

# 4. MaxAbsScaler
st.markdown("### 4. MaxAbsScaler")
st.markdown("""
Această metodă scalează fiecare caracteristică prin valoarea sa maximă absolută.
Formula: x_scaled = x / max(|x|)
Este utilă pentru date sparse și păstrează structura de date zero.
""")

maxabs_scaler = MaxAbsScaler()
maxabs_scaled_data = final_data_clean.copy()
maxabs_scaled_data[numeric_cols] = maxabs_scaler.fit_transform(maxabs_scaled_data[numeric_cols])

st.write("Date după MaxAbsScaler:")
st.write(maxabs_scaled_data)

# Vizualizare date scalate MaxAbs
fig, ax = plt.subplots(figsize=(12, 6))
data_to_plot = maxabs_scaled_data[numeric_cols].sample(min(100, len(maxabs_scaled_data)))
sns.boxplot(data=data_to_plot, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title('Distribuția datelor după MaxAbsScaler')
st.pyplot(fig)

# Comparație dintre toate metodele de scalare
st.markdown("### Comparație între metodele de scalare")
st.markdown("""
Fiecare metodă de scalare are avantaje și dezavantaje. Alegerea depinde de:
- Prezența outlierilor
- Distribuția datelor
- Tipul algoritmului ce va fi utilizat
- Importanța păstrării relației dintre date
""")

# Creați un exemplu de comparație pentru o singură variabilă
sample_feature = 'Heart Rate'  # Alege o caracteristică pentru comparație
comparison_df = pd.DataFrame({
    'Original': final_data_clean[sample_feature],
    'StandardScaler': std_scaled_data[sample_feature],
    'MinMaxScaler': minmax_scaled_data[sample_feature],
    'RobustScaler': robust_scaled_data[sample_feature],
    'MaxAbsScaler': maxabs_scaled_data[sample_feature]
})

st.write(f"Comparație pentru variabila '{sample_feature}':")
st.write(comparison_df.describe())

# Vizualizare comparativă
# Alternativă simplificată pentru comparația de scalare
fig, ax = plt.subplots(figsize=(14, 8))

# Plotează direct fiecare coloană
sns.boxplot(data=comparison_df, ax=ax)
ax.set_title(f'Comparație între metodele de scalare pentru {sample_feature}')
plt.xticks(rotation=45)
st.pyplot(fig)

# Histograme pentru comparație
fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=False)
comparison_df['Original'].hist(ax=axes[0])
axes[0].set_title('Original')
comparison_df['StandardScaler'].hist(ax=axes[1])
axes[1].set_title('StandardScaler')
comparison_df['MinMaxScaler'].hist(ax=axes[2])
axes[2].set_title('MinMaxScaler')
comparison_df['RobustScaler'].hist(ax=axes[3])
axes[3].set_title('RobustScaler')
comparison_df['MaxAbsScaler'].hist(ax=axes[4])
axes[4].set_title('MaxAbsScaler')
plt.tight_layout()
st.pyplot(fig)

# Salvarea datelor pentru analiză ulterioară
st.markdown("### Salvarea datelor codificate și scalate")
st.markdown("Alegeți tipul de scalare pentru a salva datele procesate pentru analiză ulterioară.")

scaling_option = st.selectbox(
    'Alegeți metoda de scalare pentru salvare:',
    ('StandardScaler', 'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'Fără scalare')
)

if st.button('Salvează datele procesate ca CSV'):
    if scaling_option == 'StandardScaler':
        std_scaled_data.to_csv('sleep_health_standardscaled.csv')
        st.success('Datele au fost salvate cu succes folosind StandardScaler!')
    elif scaling_option == 'MinMaxScaler':
        minmax_scaled_data.to_csv('sleep_health_minmaxscaled.csv')
        st.success('Datele au fost salvate cu succes folosind MinMaxScaler!')
    elif scaling_option == 'RobustScaler':
        robust_scaled_data.to_csv('sleep_health_robustscaled.csv')
        st.success('Datele au fost salvate cu succes folosind RobustScaler!')
    elif scaling_option == 'MaxAbsScaler':
        maxabs_scaled_data.to_csv('sleep_health_maxabsscaled.csv')
        st.success('Datele au fost salvate cu succes folosind MaxAbsScaler!')
    else:
        final_data_clean.to_csv('sleep_health_encoded.csv')
        st.success('Datele au fost salvate cu succes fără scalare!')