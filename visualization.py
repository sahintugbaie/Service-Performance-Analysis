import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import io
from matplotlib.figure import Figure

# Set plot style
plt.style.use('ggplot')

def plot_data_summary(data: pd.DataFrame):
    """
    Create summary visualizations for the data.
    
    Args:
        data: DataFrame containing the data
    """
    # Select numeric columns for visualization
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for visualization.")
        return
    
    # Create a correlation heatmap
    st.subheader("Correlation Heatmap")
    
    # Only include numeric columns with at least 2 unique values
    valid_cols = [col for col in numeric_cols if data[col].nunique() > 1]
    
    if len(valid_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = data[valid_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, 
                   fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns with variation for correlation analysis.")
    
    # Create histograms for numeric variables
    st.subheader("Distribution of Numeric Variables")
    
    # Create a 2x2 grid of histograms (up to 4 variables)
    cols_to_plot = numeric_cols[:min(4, len(numeric_cols))]
    
    if len(cols_to_plot) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            sns.histplot(data[col].dropna(), kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for j in range(len(cols_to_plot), 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(data[numeric_cols].describe())
    
    # Box plots for numeric variables
    st.subheader("Box Plots of Numeric Variables")
    
    if len(cols_to_plot) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Melt the data for easier plotting
        melted_data = pd.melt(data[cols_to_plot].reset_index(), 
                             id_vars=['index'], 
                             value_vars=cols_to_plot)
        
        # Create box plot
        sns.boxplot(x='variable', y='value', data=melted_data, ax=ax)
        ax.set_title('Box Plots of Numeric Variables')
        ax.set_xlabel('Variable')
        ax.set_ylabel('Value')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

def plot_beta_regression_results(data: pd.DataFrame, 
                               dependent_var: str, 
                               independent_vars: List[str], 
                               beta_results: Dict[str, Any]):
    """
    Create visualizations for beta regression results.
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of the dependent variable
        independent_vars: Names of the independent variables
        beta_results: Dictionary containing beta regression results
    """
    # Create coefficient plot
    st.subheader("Coefficient Plot")
    
    # Extract coefficients and standard errors, skipping the intercept
    coefs = beta_results['coefficients'][1:]  # Skip intercept
    std_errors = beta_results['std_errors'][1:]  # Skip intercept
    var_names = coefs.index
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot coefficients and confidence intervals
    y_pos = np.arange(len(var_names))
    ax.errorbar(x=coefs, y=y_pos, xerr=1.96*std_errors, fmt='o', capsize=5, 
               label='95% Confidence Interval')
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='Zero Effect')
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_names)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Beta Regression Coefficients with 95% Confidence Intervals')
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create actual vs. fitted plot
    st.subheader("Actual vs. Fitted Values")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Extract actual and fitted values
    actual = data[dependent_var]
    fitted = beta_results['fitted_values']
    
    # Create scatter plot
    ax.scatter(actual, fitted, alpha=0.5)
    
    # Add diagonal line
    min_val = min(min(actual), min(fitted))
    max_val = max(max(actual), max(fitted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Set labels and title
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Fitted Values')
    ax.set_title('Actual vs. Fitted Values')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create effects plots for top independent variables
    st.subheader("Effects of Independent Variables")
    
    # Find the top 3 most significant variables based on p-values
    p_values = beta_results['p_values'][1:]  # Skip intercept
    top_vars_idx = np.argsort(p_values)[:min(3, len(p_values))]
    top_vars = [independent_vars[i] for i in top_vars_idx]
    
    # Create a subplot for each top variable
    for var in top_vars:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create scatter plot
        ax.scatter(data[var], data[dependent_var], alpha=0.5)
        
        # Add smoothed trend line
        try:
            x_sorted = np.sort(data[var])
            
            # Avoid perfect multicollinearity by selecting only the variable of interest
            # and the intercept for prediction
            var_idx = independent_vars.index(var)
            coef = beta_results['coefficients'][var_idx + 1]  # +1 to skip intercept
            intercept = beta_results['coefficients'][0]
            
            # Simple linear predictor for visualization purposes
            y_pred = np.exp(intercept + coef * x_sorted) / (1 + np.exp(intercept + coef * x_sorted))
            
            ax.plot(x_sorted, y_pred, 'r-', linewidth=2)
        except Exception as e:
            st.warning(f"Could not add trend line for {var}: {str(e)}")
        
        # Set labels and title
        ax.set_xlabel(var)
        ax.set_ylabel(dependent_var)
        ax.set_title(f'Effect of {var} on {dependent_var}')
        
        plt.tight_layout()
        st.pyplot(fig)

def plot_residuals(beta_results: Dict[str, Any]):
    """
    Create residual plots for beta regression diagnostics.
    
    Args:
        beta_results: Dictionary containing beta regression results
    """
    # Create residual plots
    residuals = beta_results['residuals']
    fitted = beta_results['fitted_values']
    
    # Residuals vs. Fitted Values
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create scatter plot
    ax.scatter(fitted, residuals, alpha=0.5)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Add a smoothed trend line
    try:
        from scipy.stats import gaussian_kde
        
        # Sort the fitted values for the trend line
        sorted_idx = np.argsort(fitted)
        fitted_sorted = fitted[sorted_idx]
        residuals_sorted = residuals[sorted_idx]
        
        # Smooth the residuals
        from scipy.signal import savgol_filter
        if len(fitted_sorted) > 10:  # Only smooth if we have enough points
            window_size = min(15, len(fitted_sorted) // 2 * 2 + 1)  # Must be odd
            poly_order = min(3, window_size - 1)
            residuals_smooth = savgol_filter(residuals_sorted, window_size, poly_order)
            ax.plot(fitted_sorted, residuals_smooth, 'g-', linewidth=2)
    except Exception as e:
        # If smoothing fails, continue without it
        pass
    
    # Set labels and title
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs. Fitted Values')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Histogram of residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create histogram
    sns.histplot(residuals, kde=True, ax=ax)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Q-Q plot of residuals
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create Q-Q plot
    from scipy import stats
    stats.probplot(residuals, plot=ax)
    
    # Set title
    ax.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_clustering_results(data: pd.DataFrame, 
                           cluster_vars: List[str], 
                           clustering_results: Dict[str, Any],
                           method: str = "kmeans"):
    """
    Create visualizations for clustering results.
    
    Args:
        data: DataFrame containing the data
        cluster_vars: Names of the variables used for clustering
        clustering_results: Dictionary containing clustering results
        method: Clustering method used (kmeans or hierarchical)
    """
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clustering_results['cluster_assignments']
    
    # Convert Cluster to a categorical variable
    data_with_clusters['Cluster'] = data_with_clusters['Cluster'].astype('category')
    
    # Get number of clusters
    n_clusters = data_with_clusters['Cluster'].nunique()
    
    # Create scatter plots for pairs of clustering variables
    st.subheader("Cluster Visualization")
    
    # If we have more than one clustering variable, create pairwise scatter plots
    if len(cluster_vars) > 1:
        # Select pairs for visualization (up to 3 pairs)
        pairs = []
        for i in range(min(len(cluster_vars)-1, 2)):
            for j in range(i+1, min(len(cluster_vars), 3)):
                pairs.append((cluster_vars[i], cluster_vars[j]))
        
        # Create a scatter plot for each pair
        for var1, var2 in pairs:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create scatter plot with clusters as hue
            sns.scatterplot(x=var1, y=var2, hue='Cluster', data=data_with_clusters, 
                          palette='viridis', s=100, alpha=0.7, ax=ax)
            
            # Plot cluster centers if available
            if 'cluster_centers' in clustering_results and clustering_results['cluster_centers'] is not None:
                # Get indices of the two variables in the cluster_vars list
                var1_idx = cluster_vars.index(var1)
                var2_idx = cluster_vars.index(var2)
                
                # Extract center coordinates for the selected variables
                centers = clustering_results['cluster_centers'][:, [var1_idx, var2_idx]]
                
                # Add cluster centers to the plot
                ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', marker='X', 
                          alpha=1, label='Cluster Centers')
            
            # Set labels and title
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title(f'Clustering Results: {var1} vs {var2}')
            ax.legend(title='Cluster')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # If we only have one clustering variable, create a histplot
    elif len(cluster_vars) == 1:
        var = cluster_vars[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histplot with clusters as hue
        sns.histplot(x=var, hue='Cluster', data=data_with_clusters, 
                    palette='viridis', alpha=0.7, kde=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel(var)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Clustering Results: Distribution of {var} by Cluster')
        ax.legend(title='Cluster')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Create parallel coordinates plot for all clustering variables
    st.subheader("Parallel Coordinates Plot")
    
    if len(cluster_vars) > 1:
        # Create a copy of the data with only the clustering variables and cluster assignments
        plot_data = data_with_clusters[cluster_vars + ['Cluster']].copy()
        
        # Scale the data for better visualization
        for col in cluster_vars:
            plot_data[col] = (plot_data[col] - plot_data[col].min()) / (plot_data[col].max() - plot_data[col].min())
        
        # Create parallel coordinates plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each cluster
        cluster_ids = sorted(plot_data['Cluster'].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_ids)))
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = plot_data[plot_data['Cluster'] == cluster_id]
            
            # Plot each observation in the cluster
            for _, row in cluster_data.iterrows():
                ax.plot(cluster_vars, row[cluster_vars], color=colors[i], alpha=0.3)
            
            # Plot the mean for each cluster with a thicker line
            mean_values = cluster_data[cluster_vars].mean()
            ax.plot(cluster_vars, mean_values, color=colors[i], linewidth=4, 
                   label=f'Cluster {cluster_id}')
        
        # Set labels and title
        ax.set_xticks(range(len(cluster_vars)))
        ax.set_xticklabels(cluster_vars, rotation=45)
        ax.set_title('Parallel Coordinates Plot')
        ax.legend(title='Cluster')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Create box plots for each variable by cluster
    st.subheader("Box Plots by Cluster")
    
    # Create a subplot for each clustering variable
    for var in cluster_vars:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create box plot
        sns.boxplot(x='Cluster', y=var, data=data_with_clusters, palette='viridis', ax=ax)
        
        # Add individual observations
        sns.stripplot(x='Cluster', y=var, data=data_with_clusters, 
                     color='black', alpha=0.5, jitter=True, ax=ax)
        
        # Set labels and title
        ax.set_xlabel('Cluster')
        ax.set_ylabel(var)
        ax.set_title(f'Distribution of {var} by Cluster')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # If silhouette information is available, visualize it
    if 'silhouette_widths' in clustering_results and clustering_results['silhouette_widths'] is not None:
        st.subheader("Silhouette Analysis")
        
        # Extract silhouette information
        silhouette_widths = clustering_results['silhouette_widths']
        avg_silhouette = clustering_results['avg_silhouette']
        
        # Create silhouette plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot silhouette scores for each cluster
        y_lower = 10
        for i in range(1, n_clusters + 1):
            # Get silhouette scores for the current cluster
            cluster_silhouette_vals = silhouette_widths[clustering_results['cluster_assignments'] == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = len(cluster_silhouette_vals)
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(i / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label the silhouette plots with their cluster numbers
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
            
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10
        
        # Add vertical line for average silhouette width
        ax.axvline(x=avg_silhouette, color='red', linestyle='--', 
                  label=f'Average Silhouette: {avg_silhouette:.3f}')
        
        # Set labels and title
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Analysis')
        ax.legend()
        
        # Set x-axis limits
        ax.set_xlim([-0.1, 1])
        
        plt.tight_layout()
        st.pyplot(fig)
