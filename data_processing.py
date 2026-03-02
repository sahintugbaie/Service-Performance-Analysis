import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

def load_excel(file_path: str) -> pd.DataFrame:
    """
    Load data from an Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame containing the Excel data
    """
    try:
        # Read the Excel file with explicit engine specification
        data = pd.read_excel(file_path, engine='openpyxl')
        
        # Basic data cleaning
        # Remove completely empty rows and columns
        data = data.dropna(how='all').dropna(axis=1, how='all')
        
        # Print some debug information
        print(f"Successfully loaded Excel file with {data.shape[0]} rows and {data.shape[1]} columns")
        print(f"Columns: {data.columns.tolist()}")
        
        return data
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_data(data: pd.DataFrame, dependent_var: str, is_beta: bool = False) -> bool:
    """
    Validate data for analysis.
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of the dependent variable
        is_beta: Whether to validate for beta regression
        
    Returns:
        Boolean indicating if the data is valid
    """
    # Check if dependent variable exists
    if dependent_var not in data.columns:
        return False
    
    # For beta regression, check if dependent variable is between 0 and 1
    if is_beta:
        values = data[dependent_var].dropna()
        if not all((0 < values) & (values < 1)):
            return False
    
    return True

def preprocess_data(data: pd.DataFrame, 
                   dependent_var: Optional[str], 
                   independent_vars: List[str],
                   is_clustering: bool = False) -> pd.DataFrame:
    """
    Preprocess data for analysis.
    
    Args:
        data: DataFrame containing the data
        dependent_var: Name of the dependent variable (None for clustering)
        independent_vars: Names of the independent variables or clustering variables
        is_clustering: Whether preprocessing is for clustering
        
    Returns:
        Preprocessed DataFrame ready for analysis
    """
    # Select the relevant columns
    if is_clustering:
        # For clustering, we only need the clustering variables
        selected_data = data[independent_vars].copy()
    else:
        # For regression, we need dependent and independent variables
        selected_data = data[[dependent_var] + independent_vars].copy()
    
    # Handle missing values
    # For numeric columns, fill missing values with the mean
    numeric_cols = selected_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        selected_data[col] = selected_data[col].fillna(selected_data[col].mean())
    
    # For categorical columns, fill missing values with the mode
    categorical_cols = selected_data.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        selected_data[col] = selected_data[col].fillna(selected_data[col].mode()[0])
    
    # If preprocessing for clustering, standardize the data
    if is_clustering:
        for col in selected_data.columns:
            selected_data[col] = (selected_data[col] - selected_data[col].mean()) / selected_data[col].std()
    
    return selected_data

def encode_categorical_variables(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for analysis.
    
    Args:
        data: DataFrame containing the data
        
    Returns:
        DataFrame with categorical variables encoded
    """
    # Identify categorical columns
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns
    
    # Encode each categorical column
    encoded_data = data.copy()
    for col in categorical_cols:
        # Use pandas get_dummies for one-hot encoding
        dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
        
        # Drop the original column and add the dummy columns
        encoded_data = pd.concat([encoded_data.drop(col, axis=1), dummies], axis=1)
    
    return encoded_data
