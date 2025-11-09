import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

#function loads dataset and returns data as list of row Dictionaries
def load_data(filepath):
    try:
        f = open(filepath, "r", encoding = "utf-8")
        headers = f.readline().strip().split(',')
        finalList = []
        for row in f:
            values = row.strip().split(',')
            rowDict = {headers[i]: values[i] for i in range(len(headers))}
            finalList.append(rowDict)
        f.close()
        return finalList
    except Exception as e:
        print("Error loading file", e)
        return None
    

#function to test load_data
def test_load_data():
    filepath = os.path.join("countries.csv")
    outputList = (load_data(filepath))[:1]
    if outputList[0]["Country"] == "Afghanistan":
        return 2
    return 1


#returns feature_vector for country in the row provided
def  calc_features(row):
    if (type(row) != type(dict())):
        print("Error in parameter for calc_features")
        return None
    try:
        population = float(row['Population'])
        netMigration = float(row['Net migration'])
        gdpPerCapita = float(row['GDP ($ per capita)'])
        literacy = float(row['Literacy (%)'])
        phonesPer1000 = float(row['Phones (per 1000)'])
        infantMortality = float(row['Infant mortality (per 1000 births)'])

        featureVector = np.array([
            population,
            netMigration,
            gdpPerCapita,
            literacy,
            phonesPer1000,
            infantMortality
        ], dtype=np.float64)
    except Exception as e:
        print("Error", e)
        return None

    return featureVector


def hac(features):
    n = len(features)  # Number of countries feature vectors
    #initializing final array to return
    Z = np.zeros((n - 1, 4), dtype=float)
    #initializing a matrix to store distance pairwise
    distanceMatrix = np.full((n, n), np.inf, dtype=float)
    #actually calculating distances and storing in matrix
    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(features[i] - features[j])
            distanceMatrix[i, j] = distance
            distanceMatrix[j, i] = distance
    #initially every country has own cluster so range should be n
    activeClusters = list(range(n))
    cluster_sizes = {i: 1 for i in range(n)}  # Initialize cluster sizes
    for i in range(n - 1):
        min_dist = np.inf  #initially infinity
        cluster1, cluster2 = -1, -1  #initially no clusters found
        #finding closest clusters
        for j in range(len(activeClusters)):
            for k in range(j + 1, len(activeClusters)):
                dist = distanceMatrix[activeClusters[j], activeClusters[k]]
                if (dist < min_dist) and (dist > 0.00000001):
                    min_dist = dist
                    cluster1, cluster2 = activeClusters[j], activeClusters[k]
                # Tie-breaking rule
                elif dist == min_dist:
                    if (activeClusters[j] < cluster1) or ((activeClusters[j] == cluster1) and (activeClusters[k] < cluster2)):
                        cluster1, cluster2 = activeClusters[j], activeClusters[k]
        #the smaller index should come first
        if cluster1 > cluster2:
            cluster1, cluster2 = cluster2, cluster1
        #storing results in final array to be returned
        Z[i, 0] = cluster1
        Z[i, 1] = cluster2
        Z[i, 2] = min_dist
        new_cluster_size = cluster_sizes[cluster1] + cluster_sizes[cluster2]
        Z[i, 3] = new_cluster_size
        #creating new index for new cluster
        newClusterIndex = n + i
        cluster_sizes[newClusterIndex] = new_cluster_size
        # **Expand the distance matrix** for the new cluster
        newRow = np.full((1, distanceMatrix.shape[1]), np.inf)
        distanceMatrix = np.vstack([distanceMatrix, newRow])
        newCol = np.full((distanceMatrix.shape[0], 1), np.inf)
        distanceMatrix = np.hstack([distanceMatrix, newCol])
        # Updating distance matrix for the new cluster made by cluster1 and cluster2
        for cluster in activeClusters:
            if cluster != cluster1 and cluster != cluster2:
                new_distance = min(
                    distanceMatrix[cluster1, cluster],
                    distanceMatrix[cluster2, cluster]
                )
                distanceMatrix[newClusterIndex, cluster] = new_distance
                distanceMatrix[cluster, newClusterIndex] = new_distance
        #merged clusters distances in distance matrix are now invalid
        distanceMatrix[cluster1, :] = -1
        distanceMatrix[:, cluster1] = -1
        distanceMatrix[cluster2, :] = -1
        distanceMatrix[:, cluster2] = -1
        activeClusters.remove(cluster1)
        activeClusters.remove(cluster2)
        activeClusters.append(newClusterIndex)
        #making sure no duplicates
        activeClusters = list(set(activeClusters))
    return Z


def fig_hac(Z, names):
    fig = plt.figure(figsize=(12, 8))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig


def normalize_features(features):
    features_array = np.array(features)
    colMin = np.min(features_array, axis=0)
    colMax = np.max(features_array, axis=0)
    normalFeatures = (features_array - colMin) / (colMax - colMin)
    normalFeatures_list = [np.array(vec) for vec in normalFeatures]
    return normalFeatures_list


data = load_data("countries.csv")
country_names = [row["Country"] for row in data]
features = [calc_features(row) for row in data]
features_normalized = normalize_features(features)
n = 20
Z_raw = hac(features[:n])
Z_normalized = hac(features_normalized[:n])
fig = fig_hac(Z_raw, country_names[:n])
plt.show()