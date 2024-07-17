from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors


class ClusteringModel:
    def __init__(self) -> None:
        self.PCA_Data = None

    def plot_predictions_3D(self, PCA_data):
        # Define the figure and axis for 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colormap for clusters
        cmap = colors.ListedColormap(["#0077BB", "#33BEEB", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB"])

        # Scatter plot with clusters colored by 'Clusters' column
        x = PCA_data["col1"]
        y = PCA_data["col2"]
        z = PCA_data["col3"]
        scatter = ax.scatter(x, y, z, s=30, c=PCA_data["Clusters"], marker='x', cmap=cmap)
        ax.set_title("Plot of Clusters")
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Add colorbar for the clusters
        cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])  # Adjust ticks as needed
        cbar.set_label('Cluster')

        plt.savefig("./plots/Predictions.png")

    def fit(self, PCA_data):
        self.PCA_Data = PCA_data

        # Initialize KMeans and the KElbowVisualizer directly
        visualizer = KElbowVisualizer(KMeans(), k=(1, 10))

        # Fit the visualizer to PCA data
        visualizer.fit(PCA_data)

        # Visualize the elbow plot
        self.n_clusters = visualizer.elbow_value_

    def predict(self, PCA_data):
        # Initialize Agglomerative Clustering model
        AC = AgglomerativeClustering(n_clusters=self.n_clusters)

        # Fit model and predict clusters on PCA_data
        PCA_data['Clusters'] = AC.fit_predict(PCA_data)

        # Add the Clusters feature to the original dataframe
        PCA_data['Clusters'] = PCA_data['Clusters'].values
        
        self.plot_predictions_3D(PCA_data)
