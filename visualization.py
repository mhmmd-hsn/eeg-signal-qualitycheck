import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data_cleaning import DatasetCleaner

class Visualization:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the Visualization class with a pandas DataFrame.

        Parameters:
        dataframe (pd.DataFrame): The dataset to visualize.
        """
        self.data = dataframe
        self.labels = dataframe["labels"]
        self.scaled_data, _ = DatasetCleaner.standardization(self.data.drop("labels", axis=1))

    def plot_distributions(self):
        """
        Plot the distribution of each numerical feature in the dataset.
        """
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols].hist(figsize=(15, 10), bins=30, edgecolor='black')
        plt.suptitle('Distributions of Numerical Features', fontsize=16)
        plt.show()

    def plot_correlations(self):
        """
        Plot a heatmap of correlations between numerical features.
        """
        plt.figure(figsize=(12, 8))
        corr_matrix = self.data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.show()

    def plot_pairwise_relationships(self, hue="labels"):
        """
        Plot pairwise relationships between numerical features.

        Parameters:
        hue (str): Column name to use for color encoding (optional).
        """
        sns.pairplot(self.data, hue=hue, diag_kind='kde', height=2.5)
        plt.suptitle('Pairwise Relationships', y=1.02, fontsize=16)
        plt.show()

    def plot_feature_vs_target(self):
        """
        Plot each numerical feature against the target variable.

        Parameters:
        target_column (str): The name of the target variable column.
        """
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col != "target":
                plt.figure(figsize=(8, 5))
                sns.scatterplot(data=self.data, x=col, y="target", alpha=0.7)
                plt.title(f'{col} vs {"target"}')
                plt.xlabel(col)
                plt.ylabel("target")
                plt.grid(True)
                plt.show()



    def plot_distributions_by_label(self):
        """
        Plot the distribution of each feature with respect to each label in the specified column.

        Parameters:
        label_column (str): The name of the column containing the labels.
        """
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        unique_labels = self.data["labels"].unique()

        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            for label in unique_labels:
                subset = self.data[self.data["labels"] == label]
                sns.kdeplot(subset[col], label=str(label), fill=True, alpha=0.5)
            plt.title(f'Distribution of {col} by {"labels"}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend(title="target")
            plt.grid(True)
            plt.show()


    def apply_pca(self, n_components=3):
        """Apply PCA to reduce the data to 3D or 2D and plot the result."""

        if n_components == 3:
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(self.scaled_data)
            self._plot_3d(reduced_data, title="PCA")
        if n_components == 2:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(self.scaled_data)
            self._plot_2d(reduced_data, title="PCA (2D)")
        else:
            print("Visialization is only available for 2D and 3D")

    def apply_tsne(self, perplexity=10, random_state=42, n_components=3):

        """Apply t-SNE to reduce the data to 3D or 2D and plot the result."""

        if n_components == 3:
            tsne = TSNE(n_components= n_components, perplexity=perplexity, random_state=random_state)
            reduced_data = tsne.fit_transform(self.scaled_data)
            self._plot_3d(reduced_data, title="t-SNE")
        if n_components == 2:
            tsne = TSNE(n_components= n_components, perplexity=perplexity, random_state=random_state)
            reduced_data = tsne.fit_transform(self.scaled_data)
            self._plot_2d(reduced_data, title="t-SNE (2D)")
        else: 
            print("Visialization is only available for 2D and 3D")

    def _plot_2d(self, reduced_data, title):
        """Helper function to plot 2D data."""
        plt.figure()
        scatter = plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=self.labels,
            cmap="viridis",
            marker="o"
        )
        legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
        plt.gca().add_artist(legend1)
        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.show()

    def _plot_3d(self, reduced_data, title):
        """Helper function to plot 3D data."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1], 
            reduced_data[:, 2], 
            c=self.labels, 
            cmap='viridis', 
            marker='o'
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        plt.show()


if __name__ == "__main__":
    df = pd.read_csv('selected_eeg_features_with_labels.csv')
    viz = Visualization(df)
    # viz.plot_distributions() 
    # viz.plot_pairwise_relationships() 
    viz.apply_pca(n_components=2)
    # viz.plot_distributions_by_label() 
    # viz.plot_feature_vs_target(target_column='labels') 
