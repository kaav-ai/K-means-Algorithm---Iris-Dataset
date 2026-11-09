import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("iris.csv")

#INFO OF THE DATA FRAME
print(df.info())

columns_pair_characteristics = [
    ['sepal.length', 'sepal.width'],
    ['sepal.length', 'petal.length'],
    ['sepal.length', 'petal.width'],
    ['sepal.width', 'petal.length'],
    ['sepal.width', 'petal.width'],
    ['petal.length', 'petal.width']
]

output = []
asignaciones = []
centroides = []

for sc in columns_pair_characteristics:
    # Extract feature pair
    x = df[sc]
    scaler = StandardScaler()
    # Standardize feature values (mean=0, std=1)
    X_scaled = scaler.fit_transform(x)
    
    for v in range(2, 6):
        # Initialize K-Means model
        kmeans = KMeans(n_clusters=v, random_state=42, n_init=10)
        # Train model on scaled features
        kmeans.fit(X_scaled)
        
        # Plot data points colored by cluster assignment
        plt.figure(figsize=(7, 5))
        plt.scatter(
            X_scaled[:, 0],
            X_scaled[:, 1],
            c=kmeans.labels_,
            cmap="viridis",
            label="Flores"
        )

        # Plot cluster centroids on top of data points
        plt.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            marker="X",
            s=200,
            c="red",
            label="Centroides"
        )

        #Set plot labels and title
        plt.xlabel(sc[0])
        plt.ylabel(sc[1])
        plt.title(f"K-means: {sc[0]} vs {sc[1]} (k={v})")
        plt.legend()
        plt.savefig(f"graphic_pairs/{sc[0]}_vs_{sc[1]}_cluster{v}.png")
        plt.show()
        
        #Save cluster assignments for each sample
        for i, grupo in enumerate(kmeans.labels_):
            asignaciones.append({
                "Caracteristicas": sc,
                "K": v,
                "Flor": i,
                "Grupo": grupo
            })
            
        # Save centroid coordinates for each cluster
        for i, center in enumerate(kmeans.cluster_centers_):
            centroides.append({
                "Caracteristicas": sc,
                "K": v,
                "Grupo": i,
                "Centroide_1": center[0],
                "Centroide_2": center[1]
            })
            
        # Calculate cluster metrics
        inercia = kmeans.inertia_
        silueta = silhouette_score(X_scaled, kmeans.labels_)
        
        # Store metrics per feature pair and cluster count
        output.append({
            "Caracteristicas": sc,
            "K": v,
            "Inercia": inercia,
            "Silueta": silueta
        })
        
# Convert result lists into Pandas DataFrames
df_results = pd.DataFrame(output)
df_asignaciones = pd.DataFrame(asignaciones)
df_centroides = pd.DataFrame(centroides)

# Plot Elbow Method for each feature pair
for sc in columns_pair_characteristics:
    datos = df_results[
        df_results["Caracteristicas"].apply(lambda c: c == sc)
    ]

    plt.plot(
        datos["K"],
        datos["Inercia"],
        marker="o",
        label=f"{sc[0]} vs {sc[1]}"
    )

plt.xlabel("Número de clusters (K)")
plt.ylabel("Inercia")
plt.title("Método del codo")
plt.xticks([2, 3, 4, 5])
plt.legend()
plt.grid(True)
plt.savefig(f"elbow_method/elbow_method.png")
plt.show()
    
# Export evaluation metrics to CSV
df_results.to_csv("iris_result.csv", index=False)