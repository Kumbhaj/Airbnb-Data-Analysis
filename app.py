import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Import custom modules
from data_processing import (
    load_data, 
    clean_data, 
    handle_missing_values, 
    detect_outliers,
    handle_outliers,
    check_data_integrity
)
from visualization import (
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_price_heatmap,
    plot_correlation_matrix,
    plot_price_by_neighborhood,
    plot_amenities_wordcloud,
    plot_availability_calendar
)
from analysis import (
    get_summary_statistics,
    perform_feature_engineering,
    identify_patterns,
    calculate_derived_metrics
)
from utils import (
    format_summary_statistics,
    display_dataset_info
)

# Set page configuration
st.set_page_config(
    page_title="Airbnb Data Analysis Project",
    page_icon="🏠",
    layout="wide"
)

# Main app title and description
st.title("🏠 Airbnb Data Analysis Project")
st.markdown("""
This application performs comprehensive analysis on Airbnb listing data, including data cleaning, 
feature engineering, statistical analysis, and visualization of patterns and trends.

### How to use:
1. Upload an Airbnb dataset (CSV format)
2. Explore the different analysis tabs
3. Interact with the visualizations to gain insights

*Created by: A 3rd Year BTech Computer Science Student*
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
upload_section = st.sidebar.container()

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'outliers_handled' not in st.session_state:
    st.session_state.outliers_handled = None
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = "Original"

# Data upload section
with upload_section:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Example dataset option
    use_example = st.checkbox("Use example dataset", value=False)
    
    if use_example:
        try:
            # Use our local sample dataset
            st.info("Loading example dataset...")
            sample_data_path = "data/sample_airbnb_listings.csv"
            st.session_state.data = load_data(sample_data_path)
            st.success("Example dataset loaded!")
        except Exception as e:
            st.error(f"Error loading example dataset: {e}")
            st.info("Please upload your own Airbnb dataset.")
            
    elif uploaded_file is not None:
        try:
            # Load user-provided data
            st.session_state.data = load_data(uploaded_file)
            st.success("Dataset successfully loaded!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# Dataset selection in sidebar (only when data is loaded)
if st.session_state.data is not None:
    dataset_options = ["Original"]
    
    if st.session_state.cleaned_data is not None:
        dataset_options.append("Cleaned")
    
    if st.session_state.processed_data is not None:
        dataset_options.append("Processed (with Feature Engineering)")
        
    if st.session_state.outliers_handled is not None:
        dataset_options.append("Outliers Handled")
    
    st.session_state.current_dataset = st.sidebar.radio(
        "Select Dataset View:", 
        dataset_options
    )

# Navigation options in sidebar
if st.session_state.data is not None:
    selected_page = st.sidebar.radio(
        "Analysis Steps",
        ["Data Overview", 
         "Data Cleaning", 
         "Feature Engineering",
         "Statistical Analysis",
         "Pattern Identification",
         "Outlier Analysis",
         "Visualization Dashboard"]
    )
else:
    st.info("Please upload a dataset or use the example dataset to begin.")
    st.stop()

# Function to get current active dataset based on selection
def get_active_dataset():
    if st.session_state.current_dataset == "Original":
        return st.session_state.data
    elif st.session_state.current_dataset == "Cleaned":
        return st.session_state.cleaned_data
    elif st.session_state.current_dataset == "Processed (with Feature Engineering)":
        return st.session_state.processed_data
    elif st.session_state.current_dataset == "Outliers Handled":
        return st.session_state.outliers_handled
    else:
        return st.session_state.data

# Content area based on navigation selection
if selected_page == "Data Overview":
    st.header("📊 Data Overview")
    
    # Get current dataset
    df = get_active_dataset()
    
    # Display dataset information
    display_dataset_info(df)
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))
    
    # Column information
    st.subheader("Columns and Data Types")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        'Missing (%)': (df.isnull().sum().values / len(df) * 100).round(2),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info)
    
    # Show summary statistics for numerical columns
    st.subheader("Summary Statistics (Numerical Columns)")
    st.dataframe(df.describe().transpose())

elif selected_page == "Data Cleaning":
    st.header("🧹 Data Cleaning")
    
    df = get_active_dataset()
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    
    # Visualize missing values
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if len(missing_values) > 0:
        # Show missing values count
        st.write("Columns with missing values:")
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Count': missing_values.values,
            'Missing Percentage': (missing_values.values / len(df) * 100).round(2)
        }).sort_values(by='Missing Count', ascending=False)
        st.dataframe(missing_df)
        
        # Plot missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_df.sort_values('Missing Count').plot(
            kind='barh', 
            x='Column', 
            y='Missing Percentage', 
            ax=ax, 
            color='salmon'
        )
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values by Column')
        st.pyplot(fig)
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        st.write("Choose how to handle missing values for each column:")
        
        # For demonstration, handle a few important columns
        cols_to_clean = st.multiselect(
            "Select columns to clean:", 
            options=missing_values.index.tolist(),
            default=missing_values.index.tolist()[:3]  # Default select first 3
        )
        
        if st.button("Apply Missing Value Handling"):
            with st.spinner("Handling missing values..."):
                cleaned_data = handle_missing_values(df.copy(), cols_to_clean)
                st.session_state.cleaned_data = cleaned_data
                st.success("Missing values handled! Check the 'Cleaned' dataset view.")
    else:
        st.success("No missing values in the dataset!")
    
    # Check for duplicates
    st.subheader("Duplicate Records")
    duplicate_count = df.duplicated().sum()
    
    if duplicate_count > 0:
        st.warning(f"Found {duplicate_count} duplicate records ({duplicate_count/len(df)*100:.2f}% of the dataset)")
        if st.button("Remove Duplicates"):
            if st.session_state.cleaned_data is not None:
                cleaned_data = st.session_state.cleaned_data.copy()
            else:
                cleaned_data = df.copy()
            
            cleaned_data = cleaned_data.drop_duplicates()
            st.session_state.cleaned_data = cleaned_data
            st.success(f"Removed {duplicate_count} duplicate records!")
    else:
        st.success("No duplicate records found!")
    
    # Data integrity check
    st.subheader("Data Integrity Check")
    
    if st.button("Run Data Integrity Check"):
        data_integrity_issues = check_data_integrity(df)
        
        if data_integrity_issues:
            st.warning("Data integrity issues found:")
            for issue in data_integrity_issues:
                st.write(f"- {issue}")
        else:
            st.success("No data integrity issues found!")

elif selected_page == "Feature Engineering":
    st.header("🔧 Feature Engineering")
    
    # Get current dataset
    if st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.data.copy()
    
    st.write("Feature engineering involves creating new features from existing data to improve analysis and insights.")
    
    # Feature engineering options
    st.subheader("Feature Engineering Options")
    
    # Show options based on available columns
    feature_options = []
    
    # Check if price column exists
    if 'price' in df.columns:
        feature_options.append("Price Category (Budget, Mid-range, Luxury)")
        
    # Check if review-related columns exist
    if 'review_scores_rating' in df.columns:
        feature_options.append("Review Score Category")
        
    # Check if room type exists
    if 'room_type' in df.columns:
        feature_options.append("Room Type Features")
        
    # Check if location columns exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        feature_options.append("Distance to Center")
        
    # Check if neighborhood column exists
    if 'neighbourhood_cleansed' in df.columns or 'neighbourhood' in df.columns:
        feature_options.append("Neighborhood Features")
        
    # Check if availability columns exist
    if 'availability_365' in df.columns:
        feature_options.append("Availability Features")
    
    # Add more generic options
    feature_options.extend([
        "Text Length Features",
        "Amenities Count",
        "Host Features"
    ])
    
    selected_features = st.multiselect(
        "Select Features to Engineer:",
        options=feature_options,
        default=feature_options[:3]  # Default to first 3 options
    )
    
    if st.button("Apply Feature Engineering"):
        with st.spinner("Engineering features..."):
            processed_data = perform_feature_engineering(df, selected_features)
            st.session_state.processed_data = processed_data
            st.success("Feature engineering completed! Check the 'Processed' dataset view.")
            
            # Show the new features
            if st.session_state.processed_data is not None:
                new_cols = [col for col in st.session_state.processed_data.columns if col not in df.columns]
                if new_cols:
                    st.subheader("Newly Created Features:")
                    st.write(", ".join(new_cols))
                    
                    # Show sample of new features
                    st.dataframe(st.session_state.processed_data[new_cols].head(5))

elif selected_page == "Statistical Analysis":
    st.header("📈 Statistical Analysis")
    
    # Get current dataset
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data.copy()
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.data.copy()
    
    # Compute and display summary statistics
    st.subheader("Summary Statistics")
    
    # Select columns for analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_cols = st.multiselect(
        "Select columns for statistical analysis:",
        options=numerical_cols,
        default=numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
    )
    
    if selected_cols:
        stats = get_summary_statistics(df, selected_cols)
        formatted_stats = format_summary_statistics(stats)
        st.dataframe(formatted_stats)
        
        # Visualization of distributions
        st.subheader("Distribution Analysis")
        
        # Select a column to visualize
        col_to_viz = st.selectbox("Select column to visualize:", selected_cols)
        
        # Distribution plot
        fig = plot_numerical_distribution(df, col_to_viz)
        st.plotly_chart(fig)
        
        # Box plot for selected columns
        st.subheader("Box Plots - Identify Potential Outliers")
        fig = px.box(df, y=selected_cols)
        st.plotly_chart(fig)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        # Calculate correlation matrix
        if len(selected_cols) > 1:
            correlation_matrix = df[selected_cols].corr()
            
            # Plot correlation heatmap
            fig = plot_correlation_matrix(correlation_matrix)
            st.plotly_chart(fig)
            
            # Display strongest correlations
            st.subheader("Strongest Correlations")
            
            # Prepare correlation data
            corr_pairs = []
            for i in range(len(selected_cols)):
                for j in range(i+1, len(selected_cols)):
                    col1, col2 = selected_cols[i], selected_cols[j]
                    corr_value = correlation_matrix.loc[col1, col2]
                    corr_pairs.append((col1, col2, corr_value))
            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Display top correlations
            top_corr = pd.DataFrame(corr_pairs[:10], columns=['Variable 1', 'Variable 2', 'Correlation'])
            top_corr['Correlation'] = top_corr['Correlation'].round(3)
            st.dataframe(top_corr)
            
            # Scatter plot for top correlation
            if corr_pairs:
                top_pair = corr_pairs[0]
                st.subheader(f"Scatter Plot: {top_pair[0]} vs {top_pair[1]}")
                fig = px.scatter(df, x=top_pair[0], y=top_pair[1], opacity=0.6)
                st.plotly_chart(fig)
                
                st.write(f"Correlation coefficient: {top_pair[2]:.3f}")
                if abs(top_pair[2]) > 0.7:
                    st.write("This indicates a strong correlation.")
                elif abs(top_pair[2]) > 0.4:
                    st.write("This indicates a moderate correlation.")
                else:
                    st.write("This indicates a weak correlation.")
        else:
            st.info("Select at least two columns to perform correlation analysis.")
    else:
        st.warning("Please select at least one column for analysis.")

elif selected_page == "Pattern Identification":
    st.header("🔍 Pattern Identification")
    
    # Get current dataset
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data.copy()
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.data.copy()
    
    # Pattern identification tools
    st.write("This section identifies patterns, trends, and anomalies in the data.")
    
    # Price analysis across different dimensions
    st.subheader("Price Analysis Patterns")
    
    # Check if price column exists
    if 'price' in df.columns:
        # Price by categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add binary columns
        binary_cols = [col for col in df.columns if df[col].nunique() == 2]
        potential_groupby_cols = categorical_cols + binary_cols
        
        # Filter to columns with fewer than 30 unique values
        potential_groupby_cols = [col for col in potential_groupby_cols if df[col].nunique() < 30]
        
        if potential_groupby_cols:
            groupby_col = st.selectbox(
                "Select category to analyze price patterns:",
                options=potential_groupby_cols
            )
            
            # Display price statistics by selected category
            st.subheader(f"Price Statistics by {groupby_col}")
            
            # Calculate price stats
            price_stats = df.groupby(groupby_col)['price'].agg(['mean', 'median', 'count']).reset_index()
            price_stats = price_stats.sort_values(by='mean', ascending=False)
            price_stats['mean'] = price_stats['mean'].round(2)
            price_stats['median'] = price_stats['median'].round(2)
            
            # Rename columns
            price_stats.columns = [groupby_col, 'Mean Price', 'Median Price', 'Count']
            
            # Show table
            st.dataframe(price_stats)
            
            # Plot price by category
            fig = px.bar(
                price_stats,
                x=groupby_col,
                y='Mean Price',
                color='Count',
                title=f"Mean Price by {groupby_col}",
                height=500
            )
            st.plotly_chart(fig)
            
            # Show insights
            st.subheader("Insights")
            highest_category = price_stats.iloc[0][groupby_col]
            lowest_category = price_stats.iloc[-1][groupby_col]
            highest_price = price_stats.iloc[0]['Mean Price']
            lowest_price = price_stats.iloc[-1]['Mean Price']
            
            st.write(f"- The highest average price is for {highest_category} (${highest_price})")
            st.write(f"- The lowest average price is for {lowest_category} (${lowest_price})")
            st.write(f"- The price difference between highest and lowest categories is ${highest_price - lowest_price:.2f}")
        else:
            st.info("No suitable categorical columns found for price pattern analysis.")
    else:
        st.warning("Price column not found in the dataset.")
    
    # Time-based patterns (if available)
    st.subheader("Time-Based Patterns")
    
    # Check for date columns
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_columns:
        selected_date_col = st.selectbox("Select date column:", date_columns)
        
        # Try to convert to datetime
        try:
            # Ensure the column is in datetime format
            if df[selected_date_col].dtype != 'datetime64[ns]':
                df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
            
            # Extract year and month
            df['year'] = df[selected_date_col].dt.year
            df['month'] = df[selected_date_col].dt.month
            
            # Group by time periods
            time_group = st.radio("Group by:", ["Year", "Month"])
            
            if time_group == "Year" and 'price' in df.columns:
                yearly_data = df.groupby('year')['price'].agg(['mean', 'count']).reset_index()
                yearly_data.columns = ['Year', 'Average Price', 'Count']
                
                # Plot
                fig = px.line(
                    yearly_data, 
                    x='Year', 
                    y='Average Price',
                    markers=True,
                    title='Average Price by Year'
                )
                st.plotly_chart(fig)
                
                # Count by year
                fig = px.bar(
                    yearly_data,
                    x='Year',
                    y='Count',
                    title='Listing Count by Year'
                )
                st.plotly_chart(fig)
                
            elif time_group == "Month" and 'price' in df.columns:
                monthly_data = df.groupby('month')['price'].agg(['mean', 'count']).reset_index()
                monthly_data.columns = ['Month', 'Average Price', 'Count']
                
                # Get month names
                month_names = {
                    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                }
                monthly_data['Month Name'] = monthly_data['Month'].map(month_names)
                
                # Plot
                fig = px.line(
                    monthly_data, 
                    x='Month Name', 
                    y='Average Price',
                    markers=True,
                    title='Average Price by Month',
                    category_orders={"Month Name": [month_names[i] for i in range(1, 13)]}
                )
                st.plotly_chart(fig)
                
                # Count by month
                fig = px.bar(
                    monthly_data,
                    x='Month Name',
                    y='Count',
                    title='Listing Count by Month',
                    category_orders={"Month Name": [month_names[i] for i in range(1, 13)]}
                )
                st.plotly_chart(fig)
                
            # Identify patterns
            st.subheader("Identified Patterns")
            patterns = identify_patterns(df)
            
            for pattern in patterns:
                st.write(f"- {pattern}")
                
        except Exception as e:
            st.error(f"Error processing date column: {e}")
    else:
        st.info("No date columns found for time-based pattern analysis.")
    
    # Geographical patterns (if lat/long available)
    st.subheader("Geographical Patterns")
    
    if 'latitude' in df.columns and 'longitude' in df.columns and 'price' in df.columns:
        st.write("Map of listing prices by location:")
        
        # Sample if dataset is too large
        map_data = df.copy()
        if len(map_data) > 1000:
            map_data = map_data.sample(1000, random_state=42)
        
        # Create map
        fig = px.scatter_mapbox(
            map_data, 
            lat="latitude", 
            lon="longitude", 
            color="price",
            size="price",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15,
            zoom=10,
            mapbox_style="open-street-map",
            title="Listing Prices by Location"
        )
        st.plotly_chart(fig)
        
        st.write("This map shows the geographic distribution of Airbnb listings colored by price.")
        st.write("Look for patterns such as higher prices in certain neighborhoods or near attractions.")
    else:
        st.info("Location data (latitude/longitude) not available for geographical pattern analysis.")

elif selected_page == "Outlier Analysis":
    st.header("🔎 Outlier Analysis")
    
    # Get current dataset
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data.copy()
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.data.copy()
    
    st.write("This section detects and handles outliers in numerical variables.")
    
    # Select columns for outlier detection
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    selected_cols = st.multiselect(
        "Select columns for outlier detection:",
        options=numerical_cols,
        default=['price'] if 'price' in numerical_cols else numerical_cols[:1]
    )
    
    if not selected_cols:
        st.warning("Please select at least one numerical column for outlier detection.")
        st.stop()
    
    # Outlier detection method
    detection_method = st.radio(
        "Select outlier detection method:",
        ["Z-Score", "IQR (Interquartile Range)"]
    )
    
    # Detect outliers
    if st.button("Detect Outliers"):
        outliers_info = detect_outliers(df, selected_cols, method=detection_method)
        
        # Display outlier information
        st.subheader("Outlier Detection Results")
        
        for col, outliers in outliers_info.items():
            count = outliers['count']
            percentage = outliers['percentage']
            
            st.write(f"**{col}**: {count} outliers detected ({percentage:.2f}% of data)")
            
            # Visualize with box plot
            fig = px.box(df, y=col, title=f"Box Plot for {col}")
            st.plotly_chart(fig)
            
            # Show histogram with outlier thresholds
            fig = px.histogram(
                df, x=col, 
                marginal="box",
                title=f"Distribution of {col} with Outlier Thresholds"
            )
            
            # Add threshold lines
            if detection_method == "Z-Score":
                threshold = outliers['threshold']
                mean_val = df[col].mean()
                fig.add_vline(x=mean_val + threshold, line_dash="dash", line_color="red")
                fig.add_vline(x=mean_val - threshold, line_dash="dash", line_color="red")
                fig.add_annotation(x=mean_val + threshold, text="Upper threshold", showarrow=True, y=0)
                fig.add_annotation(x=mean_val - threshold, text="Lower threshold", showarrow=True, y=0)
            else:  # IQR
                q1, q3 = outliers['q1'], outliers['q3']
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                lower_bound = q1 - 1.5 * iqr
                fig.add_vline(x=upper_bound, line_dash="dash", line_color="red")
                fig.add_vline(x=lower_bound, line_dash="dash", line_color="red")
                fig.add_annotation(x=upper_bound, text="Upper threshold", showarrow=True, y=0)
                fig.add_annotation(x=lower_bound, text="Lower threshold", showarrow=True, y=0)
            
            st.plotly_chart(fig)
    
    # Outlier handling
    st.subheader("Handle Outliers")
    
    handling_method = st.radio(
        "Select outlier handling method:",
        ["Remove Outliers", "Cap Outliers", "Transform Data"]
    )
    
    if st.button("Apply Outlier Handling"):
        with st.spinner("Handling outliers..."):
            # Detect outliers first
            outliers_info = detect_outliers(df, selected_cols, method=detection_method)
            
            # Handle outliers
            df_handled = handle_outliers(df.copy(), selected_cols, outliers_info, method=handling_method)
            
            # Store in session state
            st.session_state.outliers_handled = df_handled
            
            st.success("Outliers handled successfully! Check the 'Outliers Handled' dataset view.")
            
            # Compare before and after
            st.subheader("Before vs After Outlier Handling")
            
            for col in selected_cols:
                # Create a figure with two subplots
                fig = plt.figure(figsize=(12, 5))
                
                # Before subplot
                ax1 = fig.add_subplot(121)
                sns.histplot(df[col], ax=ax1)
                ax1.set_title(f"Before: {col}")
                
                # After subplot
                ax2 = fig.add_subplot(122)
                sns.histplot(df_handled[col], ax=ax2)
                ax2.set_title(f"After: {col}")
                
                fig.tight_layout()
                st.pyplot(fig)
                
                # Statistics comparison
                stats_before = df[col].describe()
                stats_after = df_handled[col].describe()
                
                # Combine into a single dataframe
                stats_comparison = pd.DataFrame({
                    'Before': stats_before,
                    'After': stats_after
                })
                
                st.write(f"**Statistics Comparison for {col}:**")
                st.dataframe(stats_comparison)

elif selected_page == "Visualization Dashboard":
    st.header("📊 Visualization Dashboard")
    
    # Get current dataset (prefer processed or cleaned data if available)
    if st.session_state.outliers_handled is not None:
        df = st.session_state.outliers_handled.copy()
    elif st.session_state.processed_data is not None:
        df = st.session_state.processed_data.copy()
    elif st.session_state.cleaned_data is not None:
        df = st.session_state.cleaned_data.copy()
    else:
        df = st.session_state.data.copy()
    
    st.write("This dashboard provides visualizations of key patterns and insights from the Airbnb data.")
    
    # Create tabs for different visualization categories
    viz_tabs = st.tabs([
        "Price Analysis", 
        "Property Characteristics", 
        "Reviews & Ratings",
        "Location Analysis",
        "Availability Patterns"
    ])
    
    # Price Analysis Tab
    with viz_tabs[0]:
        st.subheader("Price Analysis")
        
        if 'price' in df.columns:
            # Price distribution
            st.write("### Price Distribution")
            price_fig = plot_numerical_distribution(df, 'price')
            st.plotly_chart(price_fig)
            
            # Price by categorical variables
            st.write("### Price by Categories")
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if df[col].nunique() < 20]  # Limit categories
            
            if categorical_cols:
                category_for_price = st.selectbox(
                    "Select category:",
                    options=categorical_cols
                )
                
                # Create a bar chart of average price by category
                fig = plot_price_by_neighborhood(df, category_for_price)
                st.plotly_chart(fig)
            else:
                st.info("No suitable categorical columns found for price comparison.")
            
            # Price heatmap (if we have datetime data)
            if any('date' in col.lower() for col in df.columns):
                st.write("### Price Patterns Over Time")
                
                date_cols = [col for col in df.columns if 'date' in col.lower()]
                selected_date_col = st.selectbox("Select date column:", date_cols)
                
                try:
                    # Convert to datetime if needed
                    if df[selected_date_col].dtype != 'datetime64[ns]':
                        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
                    
                    # Create price heatmap
                    fig = plot_price_heatmap(df, selected_date_col)
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Error creating price heatmap: {e}")
        else:
            st.warning("Price column not found in the dataset.")
    
    # Property Characteristics Tab
    with viz_tabs[1]:
        st.subheader("Property Characteristics")
        
        # Room type distribution
        if 'room_type' in df.columns:
            st.write("### Room Type Distribution")
            
            # Create pie chart
            room_counts = df['room_type'].value_counts()
            fig = px.pie(
                values=room_counts.values,
                names=room_counts.index,
                title="Distribution of Room Types"
            )
            st.plotly_chart(fig)
        
        # Property amenities (if available)
        if 'amenities' in df.columns:
            st.write("### Top Amenities")
            
            # Create wordcloud
            fig = plot_amenities_wordcloud(df)
            st.pyplot(fig)
        
        # Accommodates distribution (if available)
        if 'accommodates' in df.columns:
            st.write("### Accommodation Capacity")
            
            # Create histogram
            fig = px.histogram(
                df,
                x='accommodates',
                title="Distribution of Accommodation Capacity",
                marginal="box"
            )
            st.plotly_chart(fig)
            
            # Relationship between accommodates and price
            if 'price' in df.columns:
                st.write("### Price vs. Accommodation Capacity")
                
                # Create scatter plot
                fig = px.scatter(
                    df,
                    x='accommodates',
                    y='price',
                    title="Price vs. Accommodation Capacity",
                    trendline="ols"
                )
                st.plotly_chart(fig)
    
    # Reviews & Ratings Tab
    with viz_tabs[2]:
        st.subheader("Reviews & Ratings")
        
        # Check if review columns exist
        review_cols = [col for col in df.columns if 'review' in col.lower()]
        
        if review_cols:
            # Review scores distribution
            rating_cols = [col for col in review_cols if 'score' in col.lower() or 'rating' in col.lower()]
            
            if rating_cols:
                st.write("### Review Scores Distribution")
                
                selected_rating = st.selectbox(
                    "Select rating type:",
                    options=rating_cols
                )
                
                # Create histogram
                fig = px.histogram(
                    df,
                    x=selected_rating,
                    title=f"Distribution of {selected_rating}",
                    marginal="box"
                )
                st.plotly_chart(fig)
                
                # Relationship between rating and price
                if 'price' in df.columns:
                    st.write(f"### Price vs. {selected_rating}")
                    
                    # Create scatter plot
                    fig = px.scatter(
                        df,
                        x=selected_rating,
                        y='price',
                        title=f"Price vs. {selected_rating}",
                        trendline="ols"
                    )
                    st.plotly_chart(fig)
            
            # Review count analysis
            review_count_cols = [col for col in review_cols if 'number' in col.lower() or 'count' in col.lower()]
            
            if review_count_cols:
                st.write("### Review Count Analysis")
                
                selected_count = st.selectbox(
                    "Select review count column:",
                    options=review_count_cols
                )
                
                # Create histogram
                fig = px.histogram(
                    df,
                    x=selected_count,
                    title=f"Distribution of {selected_count}",
                    marginal="box"
                )
                st.plotly_chart(fig)
        else:
            st.info("No review-related columns found in the dataset.")
    
    # Location Analysis Tab
    with viz_tabs[3]:
        st.subheader("Location Analysis")
        
        # Check if location columns exist
        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.write("### Listing Locations")
            
            # Sample if dataset is too large
            map_data = df.copy()
            if len(map_data) > 1000:
                map_data = map_data.sample(1000, random_state=42)
            
            # Create map
            color_by = None
            if 'price' in df.columns:
                color_by = 'price'
            elif 'room_type' in df.columns:
                color_by = 'room_type'
            
            if color_by:
                fig = px.scatter_mapbox(
                    map_data, 
                    lat="latitude", 
                    lon="longitude", 
                    color=color_by,
                    size='price' if 'price' in df.columns else None,
                    color_continuous_scale=px.colors.cyclical.IceFire if color_by == 'price' else None,
                    size_max=15,
                    zoom=10,
                    mapbox_style="open-street-map",
                    title=f"Listing Locations Colored by {color_by}"
                )
                st.plotly_chart(fig)
            else:
                fig = px.scatter_mapbox(
                    map_data, 
                    lat="latitude", 
                    lon="longitude", 
                    zoom=10,
                    mapbox_style="open-street-map",
                    title="Listing Locations"
                )
                st.plotly_chart(fig)
            
            # Neighborhood analysis (if available)
            if 'neighbourhood' in df.columns or 'neighbourhood_cleansed' in df.columns:
                st.write("### Listings by Neighborhood")
                
                neighborhood_col = 'neighbourhood' if 'neighbourhood' in df.columns else 'neighbourhood_cleansed'
                
                # Count listings by neighborhood
                neighborhood_counts = df[neighborhood_col].value_counts().reset_index()
                neighborhood_counts.columns = ['Neighborhood', 'Count']
                neighborhood_counts = neighborhood_counts.sort_values('Count', ascending=False).head(15)
                
                # Create bar chart
                fig = px.bar(
                    neighborhood_counts,
                    x='Neighborhood',
                    y='Count',
                    title="Top 15 Neighborhoods by Listing Count"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                # Price by neighborhood (if available)
                if 'price' in df.columns:
                    st.write("### Average Price by Neighborhood")
                    
                    # Calculate average price by neighborhood
                    neighborhood_prices = df.groupby(neighborhood_col)['price'].mean().reset_index()
                    neighborhood_prices.columns = ['Neighborhood', 'Average Price']
                    neighborhood_prices = neighborhood_prices.sort_values('Average Price', ascending=False).head(15)
                    
                    # Create bar chart
                    fig = px.bar(
                        neighborhood_prices,
                        x='Neighborhood',
                        y='Average Price',
                        title="Top 15 Neighborhoods by Average Price"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig)
        else:
            st.info("Location data (latitude/longitude) not available for location analysis.")
    
    # Availability Patterns Tab
    with viz_tabs[4]:
        st.subheader("Availability Patterns")
        
        # Check if availability columns exist
        availability_cols = [col for col in df.columns if 'availability' in col.lower()]
        
        if availability_cols:
            st.write("### Availability Distribution")
            
            selected_availability = st.selectbox(
                "Select availability metric:",
                options=availability_cols
            )
            
            # Create histogram
            fig = px.histogram(
                df,
                x=selected_availability,
                title=f"Distribution of {selected_availability}",
                marginal="box"
            )
            st.plotly_chart(fig)
            
            # Relationship between availability and price
            if 'price' in df.columns:
                st.write(f"### Price vs. {selected_availability}")
                
                # Create scatter plot
                fig = px.scatter(
                    df,
                    x=selected_availability,
                    y='price',
                    title=f"Price vs. {selected_availability}",
                    trendline="ols"
                )
                st.plotly_chart(fig)
            
            # Availability calendar (if date columns exist)
            if any('date' in col.lower() for col in df.columns):
                st.write("### Availability Calendar")
                
                # Show availability calendar
                fig = plot_availability_calendar(df)
                st.plotly_chart(fig)
        else:
            st.info("Availability data not found in the dataset.")
    
    # Summary insights section
    st.subheader("Summary Insights")
    
    # Display key insights from the data
    st.write("Based on the visualizations and analysis, here are some key insights:")
    
    insights = []
    
    # Price insights
    if 'price' in df.columns:
        price_mean = df['price'].mean()
        price_median = df['price'].median()
        insights.append(f"The average price of Airbnb listings is ${price_mean:.2f}, with a median of ${price_median:.2f}.")
        
        # Price range
        price_min = df['price'].min()
        price_max = df['price'].max()
        insights.append(f"Prices range from ${price_min:.2f} to ${price_max:.2f}, showing a diverse market.")
    
    # Room type insights
    if 'room_type' in df.columns:
        most_common_room = df['room_type'].value_counts().idxmax()
        room_percentage = df['room_type'].value_counts(normalize=True).max() * 100
        insights.append(f"The most common room type is '{most_common_room}', representing {room_percentage:.1f}% of all listings.")
    
    # Rating insights
    rating_cols = [col for col in df.columns if 'score' in col.lower() or 'rating' in col.lower()]
    if rating_cols:
        for col in rating_cols[:1]:  # Just use the first rating column for insights
            avg_rating = df[col].mean()
            insights.append(f"The average {col} is {avg_rating:.2f}.")
    
    # Location insights
    if 'neighbourhood' in df.columns or 'neighbourhood_cleansed' in df.columns:
        neighborhood_col = 'neighbourhood' if 'neighbourhood' in df.columns else 'neighbourhood_cleansed'
        top_neighborhood = df[neighborhood_col].value_counts().idxmax()
        neighborhood_count = df[neighborhood_col].value_counts().max()
        neighborhood_percentage = neighborhood_count / len(df) * 100
        insights.append(f"The most popular neighborhood is '{top_neighborhood}' with {neighborhood_count} listings ({neighborhood_percentage:.1f}% of total).")
    
    # Availability insights
    availability_cols = [col for col in df.columns if 'availability' in col.lower()]
    if availability_cols:
        for col in availability_cols[:1]:  # Just use the first availability column for insights
            avg_availability = df[col].mean()
            insights.append(f"The average {col} is {avg_availability:.2f} days.")
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        st.write(f"{i}. {insight}")
    
    # Display conclusion
    st.subheader("Conclusion")
    st.write("""
    This analysis provides valuable insights into Airbnb listing characteristics, pricing patterns, and market dynamics.
    The visualizations help identify key factors affecting listing popularity and pricing, which could be useful for hosts,
    guests, and market analysts.
    
    Further analysis could involve more sophisticated modeling techniques to predict prices or occupancy rates based on
    listing characteristics, or deeper exploration of temporal patterns in pricing and availability.
    """)
