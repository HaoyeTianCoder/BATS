from sklearn.decomposition import PCA
#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd
from sklearn.manifold import TSNE

class Visual():
    def __init__(self, algorithm, number_cluster, method):
        self.method = method
        self.algorithm = algorithm
        self.number_cluster = number_cluster

    def visualize(self,  plotX):
        if self.algorithm == 'PCA':
            self.visualize_pca( plotX=plotX)
        elif self.algorithm == 'TSNE':
            self.visualize_tsne( plotX=plotX)

    def visualize_tsne(self, plotX):
        perplexity = 50

        # T-SNE with one dimension
        tsne_1d = TSNE(n_components=1, perplexity=perplexity)

        # T-SNE with two dimensions
        tsne_2d = TSNE(n_components=2, perplexity=perplexity)

        # T-SNE with three dimensions
        tsne_3d = TSNE(n_components=3, perplexity=perplexity)

        # This DataFrame holds a single dimension,built by T-SNE
        TCs_1d = pd.DataFrame(tsne_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        # This DataFrame contains two dimensions, built by T-SNE
        TCs_2d = pd.DataFrame(tsne_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        # And this DataFrame contains three dimensions, built by T-SNE
        TCs_3d = pd.DataFrame(tsne_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        TCs_1d.columns = ["TC1_1d"]

        # "TC1_2d" means: 'The first component of the components created for 2-D visualization, by T-SNE.'
        # And "TC2_2d" means: 'The second component of the components created for 2-D visualization, by T-SNE.'
        TCs_2d.columns = ["TC1_2d", "TC2_2d"]

        TCs_3d.columns = ["TC1_3d", "TC2_3d", "TC3_3d"]

        plotX = pd.concat([plotX, TCs_1d, TCs_2d, TCs_3d], axis=1, join='inner')

        plotX["dummy"] = 0

        cluster0 = plotX[plotX["Cluster"] == 0]
        cluster1 = plotX[plotX["Cluster"] == 1]
        cluster2 = plotX[plotX["Cluster"] == 2]
        cluster3 = plotX[plotX["Cluster"] == 3]


        # trace1 is for 'Cluster 0'
        trace1 = go.Scatter(
            x=cluster0["TC1_2d"],
            y=cluster0["TC2_2d"],
            mode="markers",
            name="Cluster 0",
            marker=dict(color='rgba(255, 128, 255, 0.8)'),
            text=None)

        # trace2 is for 'Cluster 1'
        trace2 = go.Scatter(
            x=cluster1["TC1_2d"],
            y=cluster1["TC2_2d"],
            mode="markers",
            name="Cluster 1",
            marker=dict(color='rgba(255, 128, 2, 0.8)'),
            text=None)

        # trace3 is for 'Cluster 2'
        trace3 = go.Scatter(
            x=cluster2["TC1_2d"],
            y=cluster2["TC2_2d"],
            mode="markers",
            name="Cluster 2",
            marker=dict(color='rgba(0, 255, 200, 0.8)'),
            text=None)

        # trace4 is for 'Cluster 3'
        trace4 = go.Scatter(
            x=cluster3["TC1_2d"],
            y=cluster3["TC2_2d"],
            mode="markers",
            name="Cluster 2",
            marker=dict(color='rgba(0, 200, 255, 0.8)'),
            text=None)

        data = [trace1, trace2, trace3, trace4]

        title = "Visualizing Clusters in Two Dimensions Using T-SNE (perplexity=" + str(perplexity) + ")"

        layout = dict(title=title,
                      xaxis=dict(title='TC1', ticklen=5, zeroline=False),
                      yaxis=dict(title='TC2', ticklen=5, zeroline=False)
                      )

        fig = dict(data=data, layout=layout)
        plot(figure_or_data=fig, filename='../fig/2D_cluster_{}.html'.format(self.method))
        # iplot(fig)

    def visualize_pca(self, plotX):
        # PCA with one principal component
        pca_1d = PCA(n_components=1)

        # PCA with two principal components
        pca_2d = PCA(n_components=2)

        # PCA with three principal components
        pca_3d = PCA(n_components=3)

        # This DataFrame holds that single principal component mentioned above
        PCs_1d = pd.DataFrame(pca_1d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        # This DataFrame contains the two principal components that will be used
        # for the 2-D visualization mentioned above
        PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        # And this DataFrame contains three principal components that will aid us
        # in visualizing our clusters in 3-D
        PCs_3d = pd.DataFrame(pca_3d.fit_transform(plotX.drop(["Cluster"], axis=1)))

        PCs_1d.columns = ["PC1_1d"]

        # "PC1_2d" means: 'The first principal component of the components created for 2-D visualization, by PCA.'
        # And "PC2_2d" means: 'The second principal component of the components created for 2-D visualization, by PCA.'
        PCs_2d.columns = ["PC1_2d", "PC2_2d"]

        PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]

        plotX = pd.concat([plotX, PCs_1d, PCs_2d, PCs_3d], axis=1, join='inner')

        plotX["dummy"] = 0

        clusters = []
        traces = []
        for i in range(self.number_cluster):
            if self.method == 'dbscan':
                i -= 1
            clusters.append(plotX[plotX["Cluster"] == i].drop(["Cluster"], axis=1))

        # init_notebook_mode(connected=True)

        colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(128,0, 0, 0.8)', 'rgba(2, 107, 155, 0.7)', 'rgba(27, 93, 3, 0.8)', 'rgba(255, 153, 0, 1)']
        for i in range(self.number_cluster):
            color = colors[i]
            trace = go.Scatter(
                x=clusters[i]["PC1_2d"],
                y=clusters[i]["PC2_2d"],
                mode="markers",
                name="Cluster "+str(i),
                marker=dict(color=color),
                text=None)
            traces.append(trace)

        title = "Visualizing Clusters in Two Dimensions Using PCA"

        layout = dict(title=title,
                      xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                      yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                      )

        fig = dict(data=traces, layout=layout)
        plot(figure_or_data=fig, filename='../fig/RQ1/2D_cluster_{}.html'.format(self.method))

        # iplot(fig)

        # 3D visual
        clusters = []
        traces = []
        for i in range(self.number_cluster):
            if self.method == 'dbscan':
                i -= 1
            clusters.append(plotX[plotX["Cluster"] == i].drop(["Cluster"], axis=1))
        colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(128,0, 0, 0.8)', 'rgba(2, 107, 155, 0.7)', 'rgba(27, 93, 3, 0.8)', 'rgba(255, 153, 0, 1)']
        for i in range(self.number_cluster):
            color = colors[i]
            trace = go.Scatter3d(
                x=clusters[i]["PC1_3d"],
                y=clusters[i]["PC2_3d"],
                z=clusters[i]["PC3_3d"],
                mode="markers",
                name="Cluster "+str(i),
                marker=dict(color=color),
                text=None)
            traces.append(trace)

        title = "Visualizing Clusters in Three Dimensions Using PCA"

        layout = dict(title=title,
                      xaxis=dict(title='PC1', ticklen=5, zeroline=False),
                      yaxis=dict(title='PC2', ticklen=5, zeroline=False)
                      )

        fig = dict(data=traces, layout=layout)
        plot(figure_or_data=fig, filename='../fig/RQ1/3D_cluster_{}.html'.format(self.method))