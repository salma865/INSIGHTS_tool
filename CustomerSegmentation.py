import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import preprocessing as prepro
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")


# Define the scoring function for Bayesian Optimization
def silhouette_scorer(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    return silhouette_score(X, cluster_labels)


# build customer_segmentation model
def customer_segmentation(data):
    data.drop(columns=['Cluster'], inplace=True)
    pca = PCA(2)
    # Transform the data
    data = pca.fit_transform(data)
    kmeans = KMeans(random_state=42)
    # Define the parameter space for Bayesian Optimization
    param_space = {
        'n_clusters': Integer(2, 10),
        'init': Categorical(['k-means++', 'random']),
        'n_init': Integer(10, 30),
        'max_iter': Integer(100, 500)
    }
    # Create the BayesSearchCV instance
    bayes_search = BayesSearchCV(
        estimator=kmeans,
        search_spaces=param_space,
        n_iter=32,
        scoring=silhouette_scorer,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    # Perform the Bayesian optimization
    bayes_search.fit(data)
    # Get the best hyperparameters and the best model
    best_params = bayes_search.best_params_
    best_kmeans = bayes_search.best_estimator_
    # Predict clusters using the best model
    label = best_kmeans.fit_predict(data)
    # Evaluate the clustering
    sil_score = silhouette_score(data, label)
    # Plotting the results
    fig = go.Figure()
    # Loop over unique cluster labels
    for i in range(best_kmeans.n_clusters):
        fig.add_trace(go.Scatter(
            x=data[label == i, 0],
            y=data[label == i, 1],
            mode='markers',
            marker=dict(size=20),
            name=f'Cluster {i}'
        ))
    fig.update_layout(
        title='K-Means Clustering',
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        legend_title='Clusters',
    )
    return fig


# df = pd.read_csv("Mall_Customers.csv")
# numerical_statistics, categorical_statistics, data = prepro.preprocessing(df)
# fig = customer_segmentation(data)
# fig.show()
