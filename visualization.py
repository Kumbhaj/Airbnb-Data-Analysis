import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import re
from collections import Counter

def plot_numerical_distribution(df, column):
    """
    Plot distribution of a numerical column
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create a histogram with kernel density estimate
    fig = px.histogram(
        df, 
        x=column,
        marginal="box",
        title=f"Distribution of {column}",
        opacity=0.7,
        histnorm="probability density",
        color_discrete_sequence=['indianred']
    )
    
    # Add a kernel density estimate on top
    try:
        kde_x, kde_y = sns.kdeplot(data=df[column].dropna()).get_lines()[0].get_data()
        fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name='KDE',
                                line=dict(color='darkblue', width=2)))
    except:
        # If KDE fails, skip it
        pass
    
    # Add layout improvements
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Density",
        showlegend=True,
        template="plotly_white"
    )
    
    # Add annotations for mean and median
    mean_val = df[column].mean()
    median_val = df[column].median()
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="green")
    fig.add_vline(x=median_val, line_dash="dash", line_color="blue")
    
    fig.add_annotation(x=mean_val, text=f"Mean: {mean_val:.2f}", showarrow=True, y=0)
    fig.add_annotation(x=median_val, text=f"Median: {median_val:.2f}", showarrow=True, y=0.05)
    
    return fig

def plot_categorical_distribution(df, column):
    """
    Plot distribution of a categorical column
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    column : str
        Column name to plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Count values
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']
    
    # Sort by count
    value_counts = value_counts.sort_values('Count', ascending=False)
    
    # Limit to top 15 categories if there are too many
    if len(value_counts) > 15:
        value_counts = value_counts.head(15)
        title = f"Top 15 Categories in {column}"
    else:
        title = f"Distribution of {column}"
    
    # Create a bar chart
    fig = px.bar(
        value_counts,
        x=column,
        y='Count',
        title=title,
        color=column,
        text='Count'
    )
    
    # Add percentage labels
    total = value_counts['Count'].sum()
    percentages = [f"{(count/total*100):.1f}%" for count in value_counts['Count']]
    
    for i, percentage in enumerate(percentages):
        fig.add_annotation(
            x=value_counts.iloc[i][column],
            y=value_counts.iloc[i]['Count'],
            text=percentage,
            showarrow=False,
            yshift=10
        )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig

def plot_price_heatmap(df, date_column):
    """
    Create a heatmap of prices over time
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    date_column : str
        Column name containing dates
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract year and month
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    
    # Group by year and month and calculate average price
    if 'price' in df.columns:
        price_by_time = df.groupby(['year', 'month'])['price'].mean().reset_index()
        
        # Create pivot table
        pivot_data = price_by_time.pivot(index='month', columns='year', values='price')
        
        # Create heatmap
        fig = px.imshow(
            pivot_data,
            labels=dict(x="Year", y="Month", color="Average Price"),
            x=pivot_data.columns,
            y=[f"{i} ({pd.to_datetime(f'2020-{i}-01').strftime('%b')})" for i in pivot_data.index],
            aspect="auto",
            color_continuous_scale="Viridis",
            title="Average Price by Year and Month"
        )
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                year = pivot_data.columns[j]
                month = pivot_data.index[i]
                try:
                    value = pivot_data.iloc[i, j]
                    if not pd.isna(value):
                        fig.add_annotation(
                            x=year,
                            y=f"{month} ({pd.to_datetime(f'2020-{month}-01').strftime('%b')})",
                            text=f"${value:.0f}",
                            showarrow=False,
                            font=dict(color="white" if value > pivot_data.mean().mean() else "black")
                        )
                except:
                    pass
        
        return fig
    else:
        # If price column doesn't exist, return an empty figure with a message
        fig = go.Figure()
        fig.update_layout(
            title="Price column not found in the dataset",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Price column not found in the dataset",
                showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig

def plot_correlation_matrix(correlation_df):
    """
    Create a heatmap of correlation matrix
    
    Parameters:
    -----------
    correlation_df : pd.DataFrame
        Correlation matrix
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create heatmap
    fig = px.imshow(
        correlation_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix",
        range_color=[-1, 1]
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return fig

def plot_price_by_neighborhood(df, category_column):
    """
    Create a bar chart of average price by neighborhood or other category
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    category_column : str
        Column name for categorization
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if 'price' not in df.columns:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Price column not found in the dataset",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Price column not found in the dataset",
                showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig
    
    # Calculate average price by category
    price_by_category = df.groupby(category_column)['price'].agg(['mean', 'count']).reset_index()
    price_by_category.columns = [category_column, 'Average Price', 'Count']
    
    # Sort by average price
    price_by_category = price_by_category.sort_values('Average Price', ascending=False)
    
    # Limit to top 15 categories if there are too many
    if len(price_by_category) > 15:
        price_by_category = price_by_category.head(15)
        title = f"Top 15 {category_column} by Average Price"
    else:
        title = f"Average Price by {category_column}"
    
    # Create bar chart
    fig = px.bar(
        price_by_category,
        x=category_column,
        y='Average Price',
        color='Count',
        title=title,
        text_auto=".2f",
        color_continuous_scale="Viridis"
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title=category_column,
        yaxis_title="Average Price ($)",
        xaxis_tickangle=-45,
        coloraxis_colorbar_title="Count"
    )
    
    return fig

def plot_amenities_wordcloud(df):
    """
    Create a word cloud visualization of amenities
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure object
    """
    if 'amenities' not in df.columns:
        # Return empty figure with message
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Amenities column not found in the dataset", 
                horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Extract amenities
    all_amenities = ' '.join(df['amenities'].dropna().astype(str))
    
    # Clean up amenities text
    all_amenities = re.sub(r'[\[\]\{\}"\'.,]', ' ', all_amenities)
    all_amenities = re.sub(r'\s+', ' ', all_amenities).strip()
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=200,
        collocations=False
    ).generate(all_amenities)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Most Common Amenities', fontsize=16)
    
    return fig

def plot_availability_calendar(df):
    """
    Create a visualization of availability patterns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Check for availability columns
    availability_cols = [col for col in df.columns if 'availability' in col.lower()]
    
    if not availability_cols:
        # Return empty figure with message
        fig = go.Figure()
        fig.update_layout(
            title="Availability data not found in the dataset",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Availability data not found in the dataset",
                showarrow=False,
                font=dict(size=20)
            )]
        )
        return fig
    
    # Select first availability column
    avail_col = availability_cols[0]
    
    # Create availability bins
    df['availability_category'] = pd.cut(
        df[avail_col],
        bins=[0, 30, 90, 180, 365],
        labels=['0-30 days', '31-90 days', '91-180 days', '181-365 days']
    )
    
    # Count by category
    availability_counts = df['availability_category'].value_counts().reset_index()
    availability_counts.columns = ['Availability', 'Count']
    
    # Sort by bin order
    bin_order = ['0-30 days', '31-90 days', '91-180 days', '181-365 days']
    availability_counts['Availability'] = pd.Categorical(
        availability_counts['Availability'],
        categories=bin_order,
        ordered=True
    )
    availability_counts = availability_counts.sort_values('Availability')
    
    # Create bar chart
    fig = px.bar(
        availability_counts,
        x='Availability',
        y='Count',
        title=f"Distribution of {avail_col}",
        color='Availability',
        text_auto=True
    )
    
    # Calculate percentages
    total = availability_counts['Count'].sum()
    for i, row in availability_counts.iterrows():
        percentage = row['Count'] / total * 100
        fig.add_annotation(
            x=row['Availability'],
            y=row['Count'],
            text=f"{percentage:.1f}%",
            showarrow=False,
            yshift=10
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Availability Range",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig
