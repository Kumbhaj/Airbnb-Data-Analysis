import pandas as pd
import numpy as np
import re
from io import StringIO

def load_data(data_source):
    """
    Load data from various sources (uploaded file, URL, etc.)
    
    Parameters:
    -----------
    data_source : file-like object or str
        Source of the data (uploaded file, URL, etc.)
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    # Check if data_source is a string (URL)
    if isinstance(data_source, str):
        if data_source.endswith('.gz'):
            df = pd.read_csv(data_source, compression='gzip')
        else:
            df = pd.read_csv(data_source)
    # Check if data_source is a file-like object
    else:
        df = pd.read_csv(data_source)
    
    # Process price column if it exists (remove $ and convert to float)
    if 'price' in df.columns:
        if df['price'].dtype == 'object':
            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Process date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    
    return df

def clean_data(df):
    """
    Perform basic data cleaning operations
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to clean
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original
    cleaned_df = df.copy()
    
    # Drop duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Convert price column if it exists
    if 'price' in cleaned_df.columns:
        if cleaned_df['price'].dtype == 'object':
            cleaned_df['price'] = cleaned_df['price'].replace('[\$,]', '', regex=True).astype(float)
    
    # Handle common date columns
    date_columns = [col for col in cleaned_df.columns if 'date' in col.lower()]
    for col in date_columns:
        try:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
        except:
            pass
    
    return cleaned_df

def handle_missing_values(df, columns_to_handle=None):
    """
    Handle missing values in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with missing values
    columns_to_handle : list, optional
        List of columns to handle missing values for. If None, handle all columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    # Make a copy to avoid modifying the original
    df_handled = df.copy()
    
    # If no columns specified, use all columns with missing values
    if columns_to_handle is None:
        columns_to_handle = df.columns[df.isnull().any()].tolist()
    
    # Handle missing values based on column type
    for col in columns_to_handle:
        # Skip if column doesn't exist
        if col not in df.columns:
            continue
            
        # Skip if no missing values
        if df[col].isnull().sum() == 0:
            continue
        
        # For numeric columns, fill with median
        if pd.api.types.is_numeric_dtype(df[col]):
            df_handled[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Get the most frequent value
            mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            df_handled[col] = df[col].fillna(mode_value)
        
        # For datetime columns, do not fill (or optionally fill with median date)
        elif pd.api.types.is_datetime64_dtype(df[col]):
            # We'll leave datetime NaT values as is
            pass
        
        # For boolean columns, fill with False
        elif pd.api.types.is_bool_dtype(df[col]):
            df_handled[col] = df[col].fillna(False)
        
        # Default case, fill with "Unknown"
        else:
            df_handled[col] = df[col].fillna("Unknown")
    
    return df_handled

def detect_outliers(df, columns, method='Z-Score', threshold=3):
    """
    Detect outliers in specified columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to detect outliers in
    columns : list
        List of columns to check for outliers
    method : str, optional
        Method to use for outlier detection ('Z-Score' or 'IQR')
    threshold : float, optional
        Threshold for Z-Score method (default is 3)
        
    Returns:
    --------
    dict
        Dictionary containing outlier information for each column
    """
    outliers_info = {}
    
    for col in columns:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Z-Score method
        if method == 'Z-Score':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = z_scores > threshold
            
            outliers_info[col] = {
                'count': outliers.sum(),
                'percentage': outliers.mean() * 100,
                'threshold': threshold,
                'indices': df.index[outliers].tolist()
            }
        
        # IQR method
        elif method == 'IQR':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            outliers_info[col] = {
                'count': outliers.sum(),
                'percentage': outliers.mean() * 100,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'indices': df.index[outliers].tolist()
            }
    
    return outliers_info

def handle_outliers(df, columns, outliers_info, method='Remove Outliers'):
    """
    Handle outliers in the DataFrame
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with outliers
    columns : list
        List of columns to handle outliers for
    outliers_info : dict
        Dictionary containing outlier information from detect_outliers()
    method : str, optional
        Method to use for handling outliers ('Remove Outliers', 'Cap Outliers', or 'Transform Data')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled outliers
    """
    # Make a copy to avoid modifying the original
    df_handled = df.copy()
    
    for col in columns:
        # Skip if column not in outliers_info
        if col not in outliers_info:
            continue
        
        # Get outlier indices
        outlier_indices = outliers_info[col]['indices']
        
        # Method 1: Remove outliers
        if method == 'Remove Outliers':
            # Create a mask for non-outlier rows
            mask = ~df_handled.index.isin(outlier_indices)
            df_handled = df_handled[mask].reset_index(drop=True)
        
        # Method 2: Cap outliers
        elif method == 'Cap Outliers':
            # For Z-Score method
            if 'threshold' in outliers_info[col]:
                mean = df[col].mean()
                std = df[col].std()
                threshold = outliers_info[col]['threshold']
                
                upper_bound = mean + threshold * std
                lower_bound = mean - threshold * std
                
                # Cap values
                df_handled.loc[df_handled[col] > upper_bound, col] = upper_bound
                df_handled.loc[df_handled[col] < lower_bound, col] = lower_bound
            
            # For IQR method
            else:
                upper_bound = outliers_info[col]['upper_bound']
                lower_bound = outliers_info[col]['lower_bound']
                
                # Cap values
                df_handled.loc[df_handled[col] > upper_bound, col] = upper_bound
                df_handled.loc[df_handled[col] < lower_bound, col] = lower_bound
        
        # Method 3: Transform data
        elif method == 'Transform Data':
            # Log transformation (for positive-only data)
            if df[col].min() > 0:
                df_handled[col] = np.log1p(df_handled[col])
            # Square root transformation (for positive-only data)
            elif df[col].min() >= 0:
                df_handled[col] = np.sqrt(df_handled[col])
            # Box-Cox transformation
            else:
                # Shift to make all values positive
                min_val = df[col].min()
                if min_val <= 0:
                    df_handled[col] = df_handled[col] - min_val + 1
                
                # Apply log transformation as a simple alternative to Box-Cox
                df_handled[col] = np.log1p(df_handled[col])
    
    return df_handled

def check_data_integrity(df):
    """
    Check for data integrity issues
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to check
        
    Returns:
    --------
    list
        List of identified data integrity issues
    """
    issues = []
    
    # Check for duplicate records
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate records ({duplicate_count/len(df)*100:.2f}% of the dataset)")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    missing_columns = missing_values[missing_values > 0]
    if not missing_columns.empty:
        issues.append(f"Found {len(missing_columns)} columns with missing values")
        for col, count in missing_columns.items():
            issues.append(f"  - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Check for potential data type issues
    for col in df.columns:
        # Check if numeric column has string values
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                pd.to_numeric(df[col])
            except:
                # Check if it looks like it should be numeric
                if any(re.search(r'\d', str(val)) for val in df[col].dropna().head(100)):
                    if not all(re.search(r'\D', str(val)) for val in df[col].dropna().head(100)):
                        issues.append(f"Column '{col}' may have mixed numeric and non-numeric values")
        
        # Check if date column has string values
        if df[col].dtype == 'object' and ('date' in col.lower() or 'time' in col.lower()):
            # Try to convert to datetime
            try:
                pd.to_datetime(df[col])
            except:
                issues.append(f"Column '{col}' appears to be a date/time column but has invalid format")
    
    # Check for potential price column issues
    if 'price' in df.columns:
        if df['price'].dtype != 'float64' and df['price'].dtype != 'int64':
            issues.append("Price column is not numeric, which may cause analysis issues")
        elif df['price'].min() < 0:
            issues.append(f"Price column has negative values (min: {df['price'].min()})")
        elif df['price'].max() > 10000:
            issues.append(f"Price column has unusually high values (max: {df['price'].max()})")
    
    return issues
