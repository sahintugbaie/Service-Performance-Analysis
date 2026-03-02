import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import traceback
import io
from data_processing import load_excel, preprocess_data, validate_data
from r_scripts import run_beta_regression, run_clustering
from visualization import (
    plot_beta_regression_results,
    plot_clustering_results,
    plot_data_summary,
    plot_residuals
)

# Set page configuration
st.set_page_config(
    page_title="Beta Regression & Clustering Analysis Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'beta_results' not in st.session_state:
    st.session_state.beta_results = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = None
if 'dependent_var' not in st.session_state:
    st.session_state.dependent_var = None
if 'independent_vars' not in st.session_state:
    st.session_state.independent_vars = None
if 'cluster_vars' not in st.session_state:
    st.session_state.cluster_vars = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3
if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = "kmeans"

# App title and description
st.title("Beta Regression & Clustering Analysis Tool")
st.markdown("""
This application allows you to perform beta regression and clustering analysis on your Excel data.
Upload your data, select variables for analysis, and get detailed visualizations and statistical summaries.
""")

# Sidebar for data upload and analysis selection
with st.sidebar:
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
    
    # Add option to use sample dataset from attached_assets
    use_sample = st.checkbox("Use sample dataset from attached_assets")
    sample_file = st.selectbox("Select sample file:", 
                              ["veriler.xlsx", "VZA Sonuç.xlsx"],
                              disabled=not use_sample)
    
    if use_sample:
        # Load the sample file directly from attached_assets
        sample_path = f"attached_assets/{sample_file}"
        try:
            # Display file details for debugging
            file_size = os.path.getsize(sample_path)
            file_details = {"FileName": sample_file, "FileSize": file_size}
            st.write(file_details)
            
            # Load data from the sample file
            st.text(f"Loading sample Excel file from: {sample_path}")
            data = load_excel(sample_path)
            
            if data is not None:
                st.session_state.data = data
                st.success("Sample data loaded successfully!")
            else:
                st.error("Failed to load sample data. Please check the Excel file.")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
            st.text("Traceback details:")
            st.text(traceback.format_exc())
    
    elif uploaded_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Display file details for debugging
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type,
                           "FileSize": uploaded_file.size}
            st.write(file_details)
            
            # Load data from the temporary file
            st.text(f"Loading Excel file from: {tmp_path}")
            data = load_excel(tmp_path)
            
            # Remove the temporary file
            os.unlink(tmp_path)
            
            if data is not None:
                st.session_state.data = data
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to load data. Please check your Excel file.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.text("Traceback details:")
            st.text(traceback.format_exc())
            # Remove the temporary file if it exists
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    st.header("Analysis Settings")
    analysis_tab = st.radio("Select Analysis", ["Data Summary", "Beta Regression", "Clustering", "Results Export"])

# Main content area
if st.session_state.data is not None:
    data = st.session_state.data
    
    if analysis_tab == "Data Summary":
        st.header("Data Summary")
        
        # Show data overview
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        st.subheader("Data Information")
        buffer = io.StringIO()
        data.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        st.subheader("Descriptive Statistics")
        st.dataframe(data.describe())
        
        # Data visualization
        st.subheader("Data Visualization")
        plot_data_summary(data)
        
    elif analysis_tab == "Beta Regression":
        st.header("Beta Regression Analysis")
        
        # Column selection for beta regression
        st.subheader("Variable Selection")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            dependent_var = st.selectbox("Select Dependent Variable (must be between 0 and 1)", 
                                        numeric_cols, 
                                        index=0 if numeric_cols else None)
        
        with col2:
            independent_vars = st.multiselect("Select Independent Variables", 
                                            [col for col in numeric_cols if col != dependent_var], 
                                            default=[col for col in numeric_cols if col != dependent_var][:min(3, len(numeric_cols)-1)])
        
        # Validate data for beta regression
        valid_data = False
        if dependent_var and independent_vars:
            valid_data = validate_data(data, dependent_var, is_beta=True)
            
            if not valid_data:
                st.warning(f"The dependent variable '{dependent_var}' must have values between 0 and 1 (exclusive) for beta regression.")
        
        # Run beta regression
        if valid_data and dependent_var and independent_vars and st.button("Run Beta Regression"):
            with st.spinner("Running Beta Regression..."):
                try:
                    # Store selected variables in session state
                    st.session_state.dependent_var = dependent_var
                    st.session_state.independent_vars = independent_vars
                    
                    # Preprocess data for beta regression
                    processed_data = preprocess_data(data, dependent_var, independent_vars)
                    
                    # Run beta regression in R
                    beta_results = run_beta_regression(processed_data, dependent_var, independent_vars)
                    
                    # Store results in session state
                    st.session_state.beta_results = beta_results
                    
                    st.success("Beta regression completed successfully!")
                except Exception as e:
                    st.error(f"Error in beta regression: {str(e)}")
        
        # Display beta regression results if available
        if st.session_state.beta_results is not None and st.session_state.dependent_var and st.session_state.independent_vars:
            st.subheader("Beta Regression Results")
            
            # Display model summary
            st.write("**Model Summary:**")
            st.text(st.session_state.beta_results['summary'])
            
            # Visualize results
            st.write("**Visualization:**")
            plot_beta_regression_results(data, st.session_state.dependent_var, 
                                         st.session_state.independent_vars, 
                                         st.session_state.beta_results)
            
            # Plot residuals
            st.write("**Residual Plots:**")
            plot_residuals(st.session_state.beta_results)
            
    elif analysis_tab == "Clustering":
        st.header("Clustering Analysis")
        
        # Column selection for clustering
        st.subheader("Variable Selection")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        cluster_vars = st.multiselect("Select Variables for Clustering", 
                                     numeric_cols, 
                                     default=numeric_cols[:min(3, len(numeric_cols))])
        
        # Clustering parameters
        col1, col2 = st.columns(2)
        with col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        
        with col2:
            clustering_method = st.selectbox("Clustering Method", ["kmeans", "hierarchical"])
        
        # Run clustering
        if cluster_vars and st.button("Run Clustering"):
            with st.spinner("Running Clustering Analysis..."):
                try:
                    # Store selected variables in session state
                    st.session_state.cluster_vars = cluster_vars
                    st.session_state.n_clusters = n_clusters
                    st.session_state.clustering_method = clustering_method
                    
                    # Preprocess data for clustering
                    processed_data = preprocess_data(data, None, cluster_vars, is_clustering=True)
                    
                    # Run clustering in R
                    clustering_results = run_clustering(processed_data, cluster_vars, 
                                                        n_clusters=n_clusters, 
                                                        method=clustering_method)
                    
                    # Store results in session state
                    st.session_state.clustering_results = clustering_results
                    
                    st.success("Clustering completed successfully!")
                except Exception as e:
                    st.error(f"Error in clustering: {str(e)}")
        
        # Display clustering results if available
        if st.session_state.clustering_results is not None and st.session_state.cluster_vars:
            st.subheader("Clustering Results")
            
            # Display cluster information
            st.write("**Cluster Information:**")
            cluster_info = st.session_state.clustering_results['cluster_info']
            st.dataframe(cluster_info)
            
            # Show which units belong to which clusters
            st.write("**Unit Cluster Assignments:**")
            
            # Create a dataframe with original data and cluster assignments
            clustered_data = data.copy()
            clustered_data['Cluster'] = st.session_state.clustering_results['cluster_assignments']
            
            # Add index column to show row numbers/unit IDs
            clustered_data.reset_index(inplace=True)
            clustered_data.rename(columns={'index': 'Unit_ID'}, inplace=True)
            clustered_data['Unit_ID'] = clustered_data['Unit_ID'] + 1  # Start from 1 instead of 0
            
            # Show cluster assignment table - ensure cluster_vars is not None
            cluster_vars_safe = st.session_state.cluster_vars if st.session_state.cluster_vars else []
            assignment_table = clustered_data[['Unit_ID', 'Cluster'] + cluster_vars_safe].copy()
            
            # Sort by cluster for better readability
            assignment_table = assignment_table.sort_values(['Cluster', 'Unit_ID'])
            
            st.dataframe(assignment_table, use_container_width=True)
            
            # Show summary of units per cluster
            st.write("**Units per Cluster Summary:**")
            cluster_summary = clustered_data.groupby('Cluster')['Unit_ID'].apply(list).reset_index()
            cluster_summary.columns = ['Cluster', 'Unit_IDs']
            cluster_summary['Unit_Count'] = cluster_summary['Unit_IDs'].apply(len)
            cluster_summary['Unit_IDs'] = cluster_summary['Unit_IDs'].apply(lambda x: ', '.join(map(str, x)))
            
            st.dataframe(cluster_summary, use_container_width=True)
            
            # Visualize clustering results
            st.write("**Visualization:**")
            plot_clustering_results(data, st.session_state.cluster_vars, 
                                   st.session_state.clustering_results, 
                                   method=st.session_state.clustering_method)
            
    elif analysis_tab == "Results Export":
        st.header("Export Analysis Results")
        
        # Beta regression results export
        st.subheader("Beta Regression Results")
        if st.session_state.beta_results is not None:
            # Convert results to CSV
            beta_results_summary = pd.DataFrame({
                'Variable': st.session_state.beta_results['coefficients'].index,
                'Coefficient': st.session_state.beta_results['coefficients'].values,
                'Std.Error': st.session_state.beta_results['std_errors'],
                'z-value': st.session_state.beta_results['z_values'],
                'p-value': st.session_state.beta_results['p_values']
            })
            
            # Download button for beta regression results
            csv_beta = beta_results_summary.to_csv(index=False)
            st.download_button(
                label="Download Beta Regression Results (CSV)",
                data=csv_beta,
                file_name="beta_regression_results.csv",
                mime="text/csv"
            )
            
            # Display summary
            st.write("**Beta Regression Summary:**")
            st.dataframe(beta_results_summary)
        else:
            st.info("No beta regression results available. Please run the analysis first.")
        
        # Clustering results export
        st.subheader("Clustering Results")
        if st.session_state.clustering_results is not None:
            # Add cluster assignments to original data
            clustered_data = data.copy()
            clustered_data['Cluster'] = st.session_state.clustering_results['cluster_assignments']
            
            # Add unit IDs for clarity
            clustered_data.reset_index(inplace=True)
            clustered_data.rename(columns={'index': 'Unit_ID'}, inplace=True)
            clustered_data['Unit_ID'] = clustered_data['Unit_ID'] + 1
            
            # Reorder columns to put Unit_ID and Cluster first
            cols = ['Unit_ID', 'Cluster'] + [col for col in clustered_data.columns if col not in ['Unit_ID', 'Cluster']]
            clustered_data = clustered_data[cols]
            
            # Download button for clustered data
            csv_clustered = clustered_data.to_csv(index=False)
            st.download_button(
                label="Download Clustered Data with Unit IDs (CSV)",
                data=csv_clustered,
                file_name="clustered_data_with_units.csv",
                mime="text/csv"
            )
            
            # Create and download cluster summary
            cluster_summary = clustered_data.groupby('Cluster')['Unit_ID'].apply(list).reset_index()
            cluster_summary.columns = ['Cluster', 'Unit_IDs']
            cluster_summary['Unit_Count'] = cluster_summary['Unit_IDs'].apply(len)
            cluster_summary['Unit_IDs'] = cluster_summary['Unit_IDs'].apply(lambda x: ', '.join(map(str, x)))
            
            csv_summary = cluster_summary.to_csv(index=False)
            st.download_button(
                label="Download Cluster Summary (CSV)",
                data=csv_summary,
                file_name="cluster_summary.csv",
                mime="text/csv"
            )
            
            # Display cluster information
            st.write("**Cluster Information:**")
            st.dataframe(st.session_state.clustering_results['cluster_info'])
            
            st.write("**Cluster Summary:**")
            st.dataframe(cluster_summary, use_container_width=True)
        else:
            st.info("No clustering results available. Please run the analysis first.")
else:
    # Display instructions when no data is loaded
    st.info("Please upload an Excel file to begin analysis.")
