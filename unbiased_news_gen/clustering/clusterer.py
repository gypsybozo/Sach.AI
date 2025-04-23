from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples # Import silhouette metrics
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer # Import Yellowbrick visualizers
import logging
from unbiased_news_gen import config
import numpy as np
import matplotlib.pyplot as plt # Import for plotting (use cautiously in scripts)
import os

# --- K-Means Tuning Functions ---

def find_optimal_kmeans_elbow(embeddings, k_range, save_path=None):
    """Finds optimal k for K-Means using the Elbow method."""
    logging.info(f"Running K-Means Elbow method for k range: {list(k_range)}...")
    if embeddings is None: return None

    try:
        # Using Yellowbrick's KElbowVisualizer
        model = KMeans(random_state=42, n_init=10)
        visualizer = KElbowVisualizer(model, k=k_range, metric='distortion', timings=False)

        visualizer.fit(embeddings) # Fit the data to the visualizer

        optimal_k = visualizer.elbow_value_
        logging.info(f"Elbow method suggests optimal k = {optimal_k}")

        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             visualizer.show(outpath=save_path, clear_figure=True) # Save the plot
             logging.info(f"Elbow plot saved to {save_path}")
        else:
            # Avoid showing plot directly in scripts, better for notebooks
            # visualizer.show()
             pass # In a script, just log the value

        plt.close() # Close the figure to prevent display issues in scripts
        return optimal_k

    except Exception as e:
        logging.error(f"Error during Elbow method visualization: {e}")
        return None


def find_optimal_kmeans_silhouette(embeddings, k_range, save_dir=None):
    """Finds optimal k for K-Means using Silhouette scores."""
    logging.info(f"Running K-Means Silhouette analysis for k range: {list(k_range)}...")
    if embeddings is None: return None

    best_k = -1
    best_score = -1 # Silhouette score ranges from -1 to 1

    # Convert sparse matrix to dense if needed (can be memory intensive!)
    # Silhouette score calculation can be slow on very large, high-dim sparse data
    if hasattr(embeddings, "toarray"):
        logging.warning("Converting sparse embeddings to dense for Silhouette analysis. This may use significant memory.")
        try:
            embeddings_dense = embeddings.toarray()
        except MemoryError:
            logging.error("MemoryError converting sparse matrix to dense for Silhouette. Consider dimensionality reduction or sampling.")
            return None
        # Optionally use TruncatedSVD for dimensionality reduction first
        # from sklearn.decomposition import TruncatedSVD
        # svd = TruncatedSVD(n_components=100) # Example: reduce to 100 dims
        # embeddings_dense = svd.fit_transform(embeddings)

    else:
        embeddings_dense = embeddings # Assume already dense (e.g., from transformer)


    silhouette_scores = {}
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_dense)

            # Skip if only one cluster is found (silhouette needs >= 2)
            if len(set(cluster_labels)) < 2:
                 logging.warning(f"Only found 1 cluster for k={k}, cannot calculate silhouette score. Skipping.")
                 continue

            score = silhouette_score(embeddings_dense, cluster_labels, metric='cosine') # Use cosine for text
            silhouette_scores[k] = score
            logging.info(f"  k={k}, Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

            # Generate and save silhouette plot for this k (optional)
            if save_dir:
                 try:
                    plt.figure() # Create a new figure for each plot
                    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick', is_fitted=True)
                    visualizer.fit(embeddings_dense) # Fit the visualizer
                    plot_path = os.path.join(save_dir, f"silhouette_k_{k}.png")
                    os.makedirs(save_dir, exist_ok=True)
                    visualizer.show(outpath=plot_path, clear_figure=True)
                    plt.close() # Close the figure
                 except Exception as plot_e:
                     logging.error(f"Failed to generate/save silhouette plot for k={k}: {plot_e}")
                     plt.close() # Ensure plot is closed even on error

        except Exception as e:
            logging.error(f"Error during Silhouette calculation for k={k}: {e}")
            continue # Continue to next k

    logging.info(f"Silhouette analysis suggests optimal k = {best_k} (Score: {best_score:.4f})")

    # Plot overall silhouette scores vs k (optional)
    if save_dir and silhouette_scores:
        try:
            plt.figure()
            plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
            plt.xlabel("Number of clusters (k)")
            plt.ylabel("Average Silhouette Score")
            plt.title("Silhouette Score vs. Number of Clusters")
            plt.grid(True)
            overall_plot_path = os.path.join(save_dir, "silhouette_scores_vs_k.png")
            plt.savefig(overall_plot_path)
            logging.info(f"Overall silhouette scores plot saved to {overall_plot_path}")
            plt.close()
        except Exception as plot_e:
            logging.error(f"Failed to save overall silhouette scores plot: {plot_e}")
            plt.close()


    return best_k

# --- DBSCAN Tuning (Placeholder) ---
def tune_dbscan_parameters(embeddings):
    """Placeholder for DBSCAN parameter tuning (more complex)."""
    # Tuning DBSCAN often involves analyzing the k-distance graph to find the 'elbow' for 'eps'
    # and considering the nature of the data for 'min_samples'.
    logging.warning("DBSCAN parameter tuning not implemented. Using default values from config.")
    # from sklearn.neighbors import NearestNeighbors
    # nn = NearestNeighbors(n_neighbors=config.DBSCAN_MIN_SAMPLES * 2, metric='cosine') # Example
    # nn.fit(embeddings)
    # distances, indices = nn.kneighbors(embeddings)
    # distances = np.sort(distances[:, config.DBSCAN_MIN_SAMPLES-1], axis=0)
    # # Plot distances and look for the 'elbow' point for eps
    # plt.plot(distances)
    # plt.xlabel("Points sorted by distance to {}-th nearest neighbor".format(config.DBSCAN_MIN_SAMPLES))
    # plt.ylabel("Epsilon (distance)")
    # plt.title("k-distance graph for DBSCAN eps tuning")
    # plt.show() # Requires manual inspection or algorithm to find elbow
    return config.DBSCAN_EPS, config.DBSCAN_MIN_SAMPLES

# --- Clustering Functions (Modified perform_kmeans_clustering) ---

def perform_kmeans_clustering(embeddings, n_clusters=None, k_range=config.CLUSTER_RANGE_KMEANS, tuning_method=config.CLUSTER_TUNING_KMEANS, plot_save_dir=None):
    """Performs K-Means, optionally tuning k first."""
    if embeddings is None:
        logging.error("Embeddings are None.")
        return None

    optimal_k = n_clusters

    # Tune k if not provided or tuning method is specified
    if optimal_k is None or tuning_method:
        logging.info(f"Tuning K-Means clusters using method: {tuning_method}")
        plot_dir = os.path.join(config.PROCESSED_DATA_DIR, 'cluster_tuning_plots') if plot_save_dir is None else plot_save_dir

        if tuning_method == 'elbow':
            optimal_k = find_optimal_kmeans_elbow(embeddings, k_range, save_path=os.path.join(plot_dir, "kmeans_elbow_plot.png"))
        elif tuning_method == 'silhouette':
             optimal_k = find_optimal_kmeans_silhouette(embeddings, k_range, save_dir=plot_dir)
        else:
            logging.warning(f"Unknown tuning method '{tuning_method}' or tuning disabled. Using default/provided n_clusters: {optimal_k}")
            if optimal_k is None:
                optimal_k = config.NUM_CLUSTERS_KMEANS # Fallback to default if still None
                logging.warning(f"Falling back to default k={optimal_k}")


    if optimal_k is None or optimal_k <= 1:
        logging.error(f"Optimal k determination failed or resulted in invalid k ({optimal_k}). Cannot perform K-Means.")
        return None

    logging.info(f"Performing final K-Means clustering with {optimal_k} clusters...")
    try:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        logging.info(f"K-Means clustering complete. Found {len(np.unique(labels))} unique cluster labels.")
        return labels, optimal_k # Return labels and the k used
    except Exception as e:
        logging.error(f"Error during final K-Means clustering: {e}")
        return None, optimal_k

def perform_dbscan_clustering(embeddings, eps=None, min_samples=None):
     # (Keep existing DBSCAN function, but maybe add tuning call)
     if eps is None or min_samples is None:
         logging.info("Tuning DBSCAN parameters (placeholder)...")
         tuned_eps, tuned_min_samples = tune_dbscan_parameters(embeddings)
         eps = tuned_eps if eps is None else eps
         min_samples = tuned_min_samples if min_samples is None else min_samples

     logging.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}...")
     # ... rest of DBSCAN logic ...
     # Convert TF-IDF to dense or use appropriate metric/algorithm if needed
     if hasattr(embeddings, "toarray"):
         logging.warning("Applying DBSCAN on potentially high-dimensional sparse TF-IDF. Consider dimensionality reduction or cosine metric.")
         # Don't convert to dense here unless memory allows, rely on algorithm/metric handling sparse data
         metric = 'cosine'
     else:
         metric = 'euclidean' # Default for dense

     try:
         dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
         labels = dbscan.fit_predict(embeddings)
         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
         n_noise_ = list(labels).count(-1)
         logging.info(f"DBSCAN clustering complete. Estimated clusters: {n_clusters_}, Noise points: {n_noise_}")
         return labels, {'eps': eps, 'min_samples': min_samples} # Return labels and params used
     except Exception as e:
        logging.error(f"Error during DBSCAN clustering: {e}")
        return None, {'eps': eps, 'min_samples': min_samples}


# --- Dispatcher Function ---
def assign_clusters(embeddings, method=config.CLUSTER_METHOD, **kwargs):
    """Dispatcher function to assign clusters based on method."""
    if method == 'kmeans':
        # Pass tuning parameters from config or kwargs
        tuning_method = kwargs.get('tuning_method', config.CLUSTER_TUNING_KMEANS)
        k_range = kwargs.get('k_range', config.CLUSTER_RANGE_KMEANS)
        n_clusters = kwargs.get('n_clusters', None) # Allow overriding config default
        return perform_kmeans_clustering(embeddings, n_clusters=n_clusters, k_range=k_range, tuning_method=tuning_method)
    elif method == 'dbscan':
        eps = kwargs.get('eps', None) # Allow overriding config default
        min_samples = kwargs.get('min_samples', None)
        return perform_dbscan_clustering(embeddings, eps=eps, min_samples=min_samples)
    else:
        logging.error(f"Unknown clustering method: {method}")
        return None, None # Return None for labels and params