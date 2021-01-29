import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score ,calinski_harabasz_score,davies_bouldin_score


def randCent(dataSet, k):
    """
    K points are randomly generated as the center of mass, where the center of mass is within the boundary of the whole data.
    """
    n = dataSet.shape[1]  # Get the dimension of the data
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = np.float(np.max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def distEclud(vecA, vecB):
    """
    Calculate the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


class biKmeans:
    def __init__(self):
        pass

    def load_data_make_blobs(self,):
        """
        Generate simulation data.
        """
        from sklearn.datasets import make_blobs  # Import method to generate simulation data
        k = 5  # Given number of clusters
        X, Y = make_blobs(n_samples=1000, n_features=2, centers=k, random_state=1)
        return X, k


    def kMeans(self, dataSet, k, distMeas=distEclud, createCent=randCent):
        """
        k-Means clustering algorithm. Return the final allocation result of k centroids and points.
        """
        m = dataSet.shape[0]  # Get the number of samples
        # Construct a cluster assignment result matrix with two columns,
        # the first column is the cluster class value to which the sample belongs,
        # and the second column is the error from the sample to the cluster centroid
        clusterAssment = np.mat(np.zeros((m, 2)))
        # 1. Initialize k centroids
        centroids = createCent(dataSet, k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                minIndex = -1
                # 2. Find the nearest centroid
                for j in range(k):
                    distJI = distMeas(centroids[j, :], dataSet[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                # 3. Update the cluster to which each row sample belongs
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2
            # print(centroids)  # Print centroid
            # 4. Update centroid
            for cent in range(k):
                ptsClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # Get all points of a given cluster
                centroids[cent, :] = np.mean(ptsClust, axis=0)  # Average along the matrix column
        return centroids, clusterAssment


    def biKmeans(self, dataSet, k, distMeas=distEclud):
        """
        Bisecting K-means Clustering Algorithm. Return the final distribution result of k centroids and points
        """
        distribution_SSE = []
        # distribution_rate = []
        m = dataSet.shape[0]
        clusterAssment = np.mat(np.zeros((m, 2)))
        # Create initial cluster centroid
        centroid0 = np.mean(dataSet, axis=0).tolist()
        centList = [centroid0]
        # Calculate the error value from each point to the centroid
        for j in range(m):
            clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
        distribution_SSE.append(sum(clusterAssment[:, 1]))
        while (len(centList) < k):
            lowestSSE = np.inf
            for i in range(len(centList)):
                # Get all the data of the current cluster
                ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
                # Perform K-Means clustering on the data of this cluster
                centroidMat, splitClustAss = self.kMeans(ptsInCurrCluster, 2, distMeas)
                sseSplit = sum(splitClustAss[:, 1])  # Calculate the sse after clustering the cluster
                sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])  # Get the sse of the remaining data set
                if (sseSplit + sseNotSplit) < lowestSSE:
                    bestCentToSplit = i
                    bestNewCents = centroidMat
                    bestClustAss = splitClustAss.copy()
                    lowestSSE = sseSplit + sseNotSplit
            # Update the cluster number 0, 1 to the number of the divided cluster and the newly added cluster
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
            bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

            # Increase centroid
            centList[bestCentToSplit] = bestNewCents[0, :]
            centList.append(bestNewCents[1, :])

            # Update cluster allocation results
            clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss

            print('cluster: {}'.format(len(centList)), end=' ')
            # print("the bestCentToSplit is: ", bestCentToSplit)
            # print("the len of bestClustAss is: ", len(bestClustAss))
            print('lowest SSE: {}'.format(float(lowestSSE)))
            distribution_SSE.append(float(lowestSSE))

        distribution_rate = [(distribution_SSE[i - 1] - distribution_SSE[i]) / distribution_SSE[i - 1] for i in range(1, len(distribution_SSE))]
        if k >= 50:
            # plt.plot(np.linspace(1, k, k), distribution_SSE)
            # plt.xlabel('Iteration', fontsize=12)
            # plt.ylabel('SSE', fontsize=12)
            # plt.savefig('../fig/RQ1/SSE_Clustering.png')
            # # plt.show()
            # plt.cla()

            plt.figure(figsize=(10, 5))
            plt.xticks(fontsize=15, )
            plt.yticks(fontsize=15, )
            plt.plot(np.linspace(2, k, k - 1), distribution_rate)
            plt.xlabel('# Clusters', fontsize=17)
            plt.ylabel('SSE Reduction Rate', fontsize=17)
            # plt.title('', fontsize=14)
            plt.savefig('../fig/RQ1/SSE_Reduction_Rate.png')
            # plt.show()
            plt.cla()

        clusters = np.array(list(map(int, np.array(clusterAssment)[:, 0])))
        return clusters