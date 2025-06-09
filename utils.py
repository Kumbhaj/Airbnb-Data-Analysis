import pandas as pd
import numpy as np
import streamlit as st

def format_summary_statistics(stats_df):
    """
    Format summary statistics for better display
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        DataFrame containing summary statistics
        
    Returns:
    --------
    pd.DataFrame
        Formatted summary statistics
    """
    # Round numeric values
    formatted_stats = stats_df.round(2)
    
    # Add additional metrics if not present
    if 'skewness' not in stats_df.columns:
        return formatted_stats
    
    # Add interpretations for skewness and kurtosis
    skew_interpretations = []
    for skew_value in formatted_stats['skewness']:
        if abs(skew_value) < 0.5:
            skew_interpretations.append("Approximately symmetric")
        elif abs(skew_value) < 1:
            skew_interpretations.append("Moderately skewed")
        else:
            if skew_value > 0:
                skew_interpretations.append("Highly right-skewed")
            else:
                skew_interpretations.append("Highly left-skewed")
    
    kurt_interpretations = []
    for kurt_value in formatted_stats['kurtosis']:
        if kurt_value < -1:
            kurt_interpretations.append("Very platykurtic (flat)")
        elif kurt_value < 0:
            kurt_interpretations.append("Platykurtic (flatter than normal)")
        elif kurt_value < 1:
            kurt_interpretations.append("Mesokurtic (normal-like)")
        else:
            kurt_interpretations.append("Leptokurtic (peaked)")
    
    # Add interpretation columns
    if len(skew_interpretations) > 0:
        formatted_stats['skewness_interpretation'] = skew_interpretations
    if len(kurt_interpretations) > 0:
        formatted_stats['kurtosis_interpretation'] = kurt_interpretations
    
    return formatted_stats

def display_dataset_info(df):
    """
    Display general information about the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to display information for
    """
    # Basic dataset information
    st.subheader("Dataset Information")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Number of Records:** {len(df):,}")
        st.write(f"**Number of Columns:** {len(df.columns):,}")
        
        # Count column types
        num_numerical = len(df.select_dtypes(include=['int64', 'float64']).columns)
        num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
        num_datetime = len(df.select_dtypes(include=['datetime64']).columns)
        num_boolean = len(df.select_dtypes(include=['bool']).columns)
        
        st.write(f"**Column Types:**")
        st.write(f"- Numerical: {num_numerical}")
        st.write(f"- Categorical: {num_categorical}")
        st.write(f"- Datetime: {num_datetime}")
        st.write(f"- Boolean: {num_boolean}")
    
    with col2:
        # Calculate missing values
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (df.shape[0] * df.shape[1])) * 100
        
        st.write(f"**Missing Values:** {missing_values:,} ({missing_percentage:.2f}%)")
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        
        # Convert to appropriate unit
        if memory_usage < 1024:
            memory_str = f"{memory_usage} bytes"
        elif memory_usage < 1024**2:
            memory_str = f"{memory_usage/1024:.2f} KB"
        elif memory_usage < 1024**3:
            memory_str = f"{memory_usage/(1024**2):.2f} MB"
        else:
            memory_str = f"{memory_usage/(1024**3):.2f} GB"
        
        st.write(f"**Memory Usage:** {memory_str}")
        
        # Duplicates
        duplicates = df.duplicated().sum()
        duplicate_percentage = (duplicates / len(df)) * 100
        
        st.write(f"**Duplicate Rows:** {duplicates:,} ({duplicate_percentage:.2f}%)")
