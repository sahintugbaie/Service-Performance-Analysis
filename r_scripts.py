import pandas as pd
import numpy as np
from typing import List, Dict, Any
import warnings
import random
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def run_beta_regression(data: pd.DataFrame, 
                        dependent_var: str, 
                        independent_vars: List[str]) -> Dict[str, Any]:
    """
    Run beta regression analysis (Python implementation in place of R).
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of the dependent variable (must be between 0 and 1)
        independent_vars: Names of the independent variables
        
    Returns:
        Dictionary containing the beta regression results
    """
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm
    
    # Extract X and y
    X = data[independent_vars]
    y = data[dependent_var]
    
    # Add constant (intercept) to independent variables
    X = sm.add_constant(X)
    
    # For beta regression, apply logit transformation to dependent variable
    # which is bounded between 0 and 1
    # y_logit = np.log(y / (1 - y))
    
    # Instead of true beta regression, we'll use OLS on the logit-transformed data
    # This is an approximation of beta regression
    
    # Apply logit transformation safely
    eps = 1e-10  # Small value to avoid log(0) or log(1)
    y_adj = y.copy()
    y_adj = y_adj.clip(eps, 1-eps)  # Bound y strictly between 0 and 1
    y_logit = np.log(y_adj / (1 - y_adj))
    
    # Fit OLS model
    model = sm.OLS(y_logit, X).fit()
    
    # Get summary as string
    summary_str = str(model.summary())
    
    # Extract coefficients and statistics
    coefficients = model.params
    std_errors = model.bse
    t_values = model.tvalues
    p_values = model.pvalues
    
    # Calculate fitted values and residuals
    fitted_logit = model.predict(X)
    fitted_values = 1 / (1 + np.exp(-fitted_logit))  # Inverse logit transform
    residuals = y - fitted_values
    
    # Calculate pseudo R-squared
    # (using McFadden's method as approximation)
    pseudo_r2 = model.rsquared
    
    # Extract log-likelihood
    loglik = model.llf
    
    # Return results as a dictionary
    results = {
        'coefficients': coefficients,
        'std_errors': np.array(std_errors),
        'z_values': np.array(t_values),  # t-values as approximation of z-values
        'p_values': np.array(p_values),
        'fitted_values': np.array(fitted_values),
        'residuals': np.array(residuals),
        'pseudo_r2': pseudo_r2,
        'loglik': loglik,
        'summary': summary_str
    }
    
    return results

def run_clustering(data: pd.DataFrame, 
                  cluster_vars: List[str], 
                  n_clusters: int = 3, 
                  method: str = "kmeans") -> Dict[str, Any]:
    """
    Run clustering analysis using scikit-learn instead of R.
    
    Args:
        data: DataFrame containing the data
        cluster_vars: Names of the variables to use for clustering
        n_clusters: Number of clusters
        method: Clustering method (kmeans or hierarchical)
        
    Returns:
        Dictionary containing the clustering results
    """
    # Extract the data for clustering
    cluster_data = data[cluster_vars].values
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # Run clustering based on the selected method
    if method == "kmeans":
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_assignments = kmeans.fit_predict(scaled_data)
        
        # K-means uses 0-indexed clusters, convert to 1-indexed to match R
        cluster_assignments = cluster_assignments + 1
        
        # Extract cluster centers
        cluster_centers = kmeans.cluster_centers_
        
        # Calculate within-cluster sum of squares
        withinss = []
        for i in range(n_clusters):
            # Find points in this cluster (using 0-indexed cluster labels)
            cluster_points = scaled_data[cluster_assignments - 1 == i]
            if len(cluster_points) > 0:
                # Calculate within-cluster sum of squares
                within_ss = np.sum(np.sum((cluster_points - cluster_centers[i]) ** 2, axis=1))
                withinss.append(within_ss)
            else:
                withinss.append(0)
        
        withinss = np.array(withinss)
        tot_withinss = np.sum(withinss)
        
        # Calculate total sum of squares
        total_center = np.mean(scaled_data, axis=0)
        total_ss = np.sum(np.sum((scaled_data - total_center) ** 2, axis=1))
        
        # Calculate between-cluster sum of squares
        betweenss = total_ss - tot_withinss
        
    else:  # hierarchical
        # Hierarchical clustering
        # Calculate distance matrix
        distance_matrix = pdist(scaled_data)
        
        # Run hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='ward')
        
        # Cut the tree to get clusters
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate cluster centers manually
        cluster_centers = np.zeros((n_clusters, len(cluster_vars)))
        for i in range(1, n_clusters + 1):
            # Find points in this cluster
            cluster_points = scaled_data[cluster_assignments == i]
            if len(cluster_points) > 0:
                # Calculate cluster center
                cluster_centers[i-1] = np.mean(cluster_points, axis=0)
        
        # Calculate within-cluster sum of squares
        withinss = []
        for i in range(1, n_clusters + 1):
            # Find points in this cluster
            cluster_points = scaled_data[cluster_assignments == i]
            if len(cluster_points) > 0:
                # Calculate within-cluster sum of squares
                within_ss = np.sum(np.sum((cluster_points - cluster_centers[i-1]) ** 2, axis=1))
                withinss.append(within_ss)
            else:
                withinss.append(0)
        
        withinss = np.array(withinss)
        tot_withinss = np.sum(withinss)
        
        # Calculate total sum of squares
        total_center = np.mean(scaled_data, axis=0)
        total_ss = np.sum(np.sum((scaled_data - total_center) ** 2, axis=1))
        
        # Calculate between-cluster sum of squares
        betweenss = total_ss - tot_withinss
    
    # Calculate cluster statistics
    cluster_stats = []
    for i in range(1, n_clusters + 1):
        cluster_indices = np.where(cluster_assignments == i)[0]
        cluster_size = len(cluster_indices)
        cluster_stats.append({
            'Cluster': i,
            'Size': cluster_size,
            'Within_SS': withinss[i-1] if i-1 < len(withinss) else 0
        })
    
    # Create a DataFrame for cluster statistics
    cluster_info = pd.DataFrame(cluster_stats)
    
    # Calculate Silhouette score
    try:
        if len(np.unique(cluster_assignments)) > 1:  # Ensure we have at least 2 clusters
            # Calculate silhouette scores for each sample
            silhouette_widths = silhouette_samples(scaled_data, cluster_assignments)
            
            # Calculate average silhouette width
            avg_silhouette = silhouette_score(scaled_data, cluster_assignments)
            
            silhouette_info = True
        else:
            silhouette_widths = None
            avg_silhouette = None
            silhouette_info = False
    except Exception as e:
        # If silhouette calculation fails, continue without it
        warnings.warn(f"Silhouette calculation failed: {str(e)}")
        silhouette_widths = None
        avg_silhouette = None
        silhouette_info = False
    
    # Return results as a dictionary
    results = {
        'cluster_assignments': cluster_assignments,
        'cluster_centers': cluster_centers,
        'cluster_info': cluster_info,
        'tot_withinss': tot_withinss,
        'betweenss': betweenss,
        'method': method
    }
    
    # Add silhouette information if available
    if silhouette_info:
        results['silhouette_widths'] = silhouette_widths
        results['avg_silhouette'] = avg_silhouette
    
    return results