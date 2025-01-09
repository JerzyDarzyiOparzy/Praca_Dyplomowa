import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Tworzenie folderu na wykresy
os.makedirs("wykresy", exist_ok=True)

# Wczytanie danych
file_path = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
raw_data = pd.read_csv(file_path)

# Zarządzanie duplikatami
num_duplicates = raw_data.duplicated().sum()
print(f"Liczba duplikatów: {num_duplicates}")

data = raw_data.drop_duplicates()

# Styl wykresów
sns.set(style="whitegrid", palette="pastel")

# ======================== Korelacja i heatmapa ======================== #
correlation_matrix = data.corr()

# Tworzenie heatmapy
plt.figure(figsize=(15, 12), dpi=300)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Tworzenie maski górnego trójkąta
sns.set(style="white")
sns.heatmap(
    correlation_matrix,
    mask=mask,
    cmap='viridis',
    annot=True,
    fmt=".2f",
    linewidths=.5,
    annot_kws={'size': 10},
)
plt.title('Correlation Heatmap', size=20)
plt.tight_layout()
plt.savefig("wykresy/heatmapa_korelacji.png", dpi=300)
plt.show()

# Korelacje z 'Diabetes_binary'
correlations = data.drop(columns=['Diabetes_binary']).corrwith(data['Diabetes_binary']).sort_values(ascending=False)
plt.figure(figsize=(10, 8), dpi=300)
sns.barplot(x=correlations.values, y=correlations.index, palette="viridis")
plt.title("Korelacje zmiennych z Diabetes_binary", fontsize=14)
plt.xlabel("Wartość korelacji", fontsize=12)
plt.ylabel("Zmienne", fontsize=12)
plt.tight_layout()
plt.savefig("wykresy/korelacje_diabetes_binary.png", dpi=300)
plt.show()

# ======================== Analiza zmiennych liczbowych ======================== #
numeric_features = ["BMI", "Age", "PhysHlth", "MentHlth", "GenHlth", "Education", "Income"]
summary = []

# Ustawienie opcji wyświetlania w Pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.colheader_justify', 'center')

# Analiza IQR i wartości odstających
for feature in numeric_features:
    feature_data = data[feature]
    Q1 = feature_data.quantile(0.25)
    Q3 = feature_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_below = feature_data[feature_data < lower_bound].count()
    outliers_above = feature_data[feature_data > upper_bound].count()

    summary.append({
        "Zmienna": feature,
        "Min": feature_data.min(),
        "Max": feature_data.max(),
        "Mediana": feature_data.median(),
        "50%": feature_data.quantile(0.5),
        "1. kwartyl (Q1)": Q1,
        "3. kwartyl (Q3)": Q3,
        "IQR": IQR,
        "Wartości odstające poniżej": outliers_below,
        "Wartości odstające powyżej": outliers_above
    })

# Konwersja do DataFrame i wyświetlenie wyników
summary_df = pd.DataFrame(summary)
print(summary_df)

# ======================== Wykres liczby wartości odstających ======================== #
outliers_df = summary_df[["Zmienna", "Wartości odstające powyżej"]].set_index("Zmienna")
outliers_above_df = outliers_df[outliers_df["Wartości odstające powyżej"] > 0]

# Tworzenie niestandardowego gradientu: czerwony -> pomarańczowy -> niebieski
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_gradient", ["red", "purple", "blue"]
)

# Normalizacja wartości dla przypisania kolorów
normalized_values = np.linspace(0, 1, len(outliers_above_df))

# Pobieranie kolorów z mapy gradientu
colors = [custom_cmap(val) for val in normalized_values]

# Tworzenie wykresu słupkowego
plt.figure(figsize=(12, 6), dpi=300)
bars = plt.bar(
    outliers_above_df.index,
    outliers_above_df["Wartości odstające powyżej"],
    color=colors,
    edgecolor="black",
    linewidth=0.7,
)

# Dodanie liczby wartości nad słupkami
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 100,
        f"{int(height)}",
        ha="center",
        fontsize=10,
        color="black",
    )

# Formatowanie wykresu
plt.title("Liczba wartości odstających", fontsize=16)
plt.xlabel("Zmienna", fontsize=14)
plt.ylabel("Liczba wartości odstających", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("wykresy/liczba_odstajacych_gradient.png", dpi=300)
plt.show()


# ======================== Rozkład zmiennych liczbowych ======================== #
for feature in numeric_features:
    plt.figure(figsize=(8, 6), dpi=300)

    # Histogram z KDE
    sns.histplot(data[feature], kde=True, color="skyblue", bins=30)

    plt.title(f"Rozkład zmiennej liczbowej: {feature}", fontsize=16)
    plt.xlabel("Wartości", fontsize=12)
    plt.ylabel("Liczność", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"wykresy/rozkład_{feature}.png", dpi=300)
    plt.show()

selected_columns = [
    "Diabetes_binary",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex"
]

# Wybór tylko wybranych zmiennych
selected_data = raw_data[selected_columns]

# Zliczanie wartości 0 i 1
binary_counts = selected_data.melt(var_name='Variable', value_name='Value').groupby(['Variable', 'Value']).size().unstack()

# Obliczanie procentowego udziału wartości 0 i 1
binary_percentages = binary_counts.div(binary_counts.sum(axis=1), axis=0) * 100

# Sortowanie według liczby wartości 0
binary_counts = binary_counts.sort_values(by=0, ascending=False)
binary_percentages = binary_percentages.loc[binary_counts.index]

# Przygotowanie wykresu poziomego
plt.figure(figsize=(14, 10), dpi=300)
ax = binary_counts.plot(kind='barh', stacked=True, color=['skyblue', 'salmon'], figsize=(14, 10), width=0.8)

# Dodanie etykiet procentowych na paskach
for i, (index, row_counts) in enumerate(binary_counts.iterrows()):
    row_percentages = binary_percentages.loc[index]
    total = sum(row_counts)
    for j, percentage in enumerate(row_percentages):
        count = row_counts[j]
        plt.text(
            count / 2 if j == 0 else total - count / 2,
            i,
            f"{percentage:.1f}%",
            va='center',
            ha='center',
            fontsize=14,
            color='black'
        )

plt.title("Rozkład procentowy zmiennych binarnych)", fontsize=16, pad=20)
plt.xlabel("Liczba", fontsize=14)
plt.ylabel("Ilość", fontsize=14)
plt.legend(["0", "1"], title="Wartość", loc='center left', bbox_to_anchor=(0.8, -0.08), ncol=2, fontsize=12, title_fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Dostosowanie wykresu, aby dotykał lewej i prawej strony
plt.xlim(0, binary_counts.sum(axis=1).max())
plt.tight_layout()
plt.savefig("wykresy/rozkład_binarnych_zmiennych.png", dpi=300)
plt.show()
