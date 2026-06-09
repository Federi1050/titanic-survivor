import seaborn as sns
import matplotlib.pyplot as plt

class Grafici:
    def plot_correlation(self, data):
        corr = data.corr(numeric_only=True)
        fig = plt.figure(figsize=(10,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice di correlazione")
        plt.show()
        return fig

    def plot_hist(self, data, col):
        fig = plt.figure(figsize=(6,4))
        plt.hist(data[col].dropna(), bins=30)
        plt.title(f"Distribuzione di {col}")
        plt.show()
        return fig

    def plot_distribution(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribuzione di {col}")
        plt.show()
        return fig

    def plot_scatter(self, data, x, y):
        fig = plt.figure(figsize=(6,4))
        sns.scatterplot(data=data, x=x, y=y)
        plt.title(f"{x} vs {y}")
        plt.show()
        return fig

    def plot_box(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.boxplot(y=data[col])
        plt.title(f"Boxplot di {col}")
        plt.show()
        return fig

    def plot_counts(self, data, col):
        fig = plt.figure(figsize=(6,4))
        sns.countplot(y=data[col])
        plt.title(f"Distribuzione di {col}")
        plt.show()
        return fig