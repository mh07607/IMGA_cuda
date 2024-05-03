import numpy as np
from sklearn.cluster import SpectralClustering

def calculate_affinity_matrix(population):
    n = len(population)
    affinity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                affinity_matrix[i, j] = 1
            else:
                affinity_matrix[i, j] = LCS(population[i], population[j]) / max(len(population[i]), len(population[j]))
    
    return affinity_matrix

def spectral_clustering(population, k):
    affinity_matrix = calculate_affinity_matrix(population)
    laplacian_matrix = np.diag(np.sum(affinity_matrix, axis=1)) - affinity_matrix
    
    # Calculate the first k smallest eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)
    indices = np.argsort(eigenvalues)[:k]
    eigenvectors_k = eigenvectors[:, indices]
    
    # Apply k-means clustering on the eigenvectors
    kmeans = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans', random_state=42)
    clusters = kmeans.fit_predict(affinity_matrix)
    
    return clusters

def LCS(sequence1, sequence2):
    n = len(sequence1)
    m = len(sequence2)
    
    # Initialize an LCS matrix
    LCS_matrix = np.zeros((n + 1, m + 1))
    
    # Fill the LCS matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if sequence1[i - 1] == sequence2[j - 1]:
                LCS_matrix[i][j] = LCS_matrix[i - 1][j - 1] + 1
            else:
                LCS_matrix[i][j] = max(LCS_matrix[i - 1][j], LCS_matrix[i][j - 1])
    
    return LCS_matrix[n][m]

# Example usage
population = [
    [8, 6, 7, 4, 5, 3, 1, 2],
    [8, 4, 6, 7, 5, 1, 3, 2],
    [8, 4, 6, 7, 5, 1, 3, 2],
    [8, 4, 6, 7, 5, 1, 3, 2][::-1]
]
k = 2

clusters = spectral_clustering(population, k)
print("Clusters:", clusters)
