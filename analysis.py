import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime

def get_summary_statistics(df, columns):
    """
    Calculate summary statistics for numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    columns : list
        List of columns to calculate statistics for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with summary statistics
    """
    # Basic statistics
    basic_stats = df[columns].describe().transpose()
    
    # Additional statistics
    additional_stats = pd.DataFrame({
        'skewness': df[columns].skew(),
        'kurtosis': df[columns].kurtosis(),
        'median': df[columns].median(),
        'missing': df[columns].isnull().sum(),
        'missing_pct': (df[columns].isnull().sum() / len(df) * 100).round(2),
        'unique': df[columns].nunique()
    })
    
    # Combine statistics
    all_stats = pd.concat([basic_stats, additional_stats], axis=1)
    
    return all_stats

def perform_feature_engineering(df, selected_features):
    """
    Perform feature engineering on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to engineer features for
    selected_features : list
        List of feature engineering options to apply
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with engineered features
    """
    # Make a copy to avoid modifying the original
    df_engineered = df.copy()
    
    # Price Category (Budget, Mid-range, Luxury)
    if 'Price Category (Budget, Mid-range, Luxury)' in selected_features and 'price' in df.columns:
        price_quantiles = df['price'].quantile([0.33, 0.67]).values
        
        def categorize_price(price):
            if price <= price_quantiles[0]:
                return 'Budget'
            elif price <= price_quantiles[1]:
                return 'Mid-range'
            else:
                return 'Luxury'
        
        df_engineered['price_category'] = df['price'].apply(categorize_price)
    
    # Review Score Category
    if 'Review Score Category' in selected_features:
        review_cols = [col for col in df.columns if 'review' in col.lower() and 'score' in col.lower()]
        
        for col in review_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_name = col.lower().replace('review_scores_', '').replace('_', '')
                
                # Create categories based on score ranges
                if df[col].max() <= 5:  # 5-point scale
                    bins = [0, 3, 4, 4.5, 5]
                    labels = ['Poor', 'Good', 'Very Good', 'Excellent']
                elif df[col].max() <= 10:  # 10-point scale
                    bins = [0, 6, 8, 9, 10]
                    labels = ['Poor', 'Good', 'Very Good', 'Excellent']
                elif df[col].max() <= 100:  # 100-point scale
                    bins = [0, 60, 80, 90, 100]
                    labels = ['Poor', 'Good', 'Very Good', 'Excellent']
                else:
                    continue
                
                df_engineered[f'{col_name}_category'] = pd.cut(
                    df[col], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
    
    # Room Type Features
    if 'Room Type Features' in selected_features and 'room_type' in df.columns:
        # One-hot encode room type
        room_type_dummies = pd.get_dummies(df['room_type'], prefix='room')
        df_engineered = pd.concat([df_engineered, room_type_dummies], axis=1)
        
        # Create binary features for private rooms
        if 'Entire home/apt' in df['room_type'].values:
            df_engineered['is_entire_home'] = df['room_type'] == 'Entire home/apt'
        
        if 'Private room' in df['room_type'].values:
            df_engineered['is_private_room'] = df['room_type'] == 'Private room'
    
    # Distance to Center
    if 'Distance to Center' in selected_features and 'latitude' in df.columns and 'longitude' in df.columns:
        # Define city centers (example: New York City)
        # Normally we would have a more sophisticated approach to determine the city
        city_centers = {
            'New York': (40.7128, -74.0060),
            'London': (51.5074, -0.1278),
            'Paris': (48.8566, 2.3522),
            'Berlin': (52.5200, 13.4050),
            'Tokyo': (35.6762, 139.6503),
            'Sydney': (-33.8688, 151.2093)
        }
        
        # Try to determine the city based on coordinates
        city = 'New York'  # Default
        mean_lat = df['latitude'].mean()
        mean_lng = df['longitude'].mean()
        
        min_distance = float('inf')
        for city_name, (city_lat, city_lng) in city_centers.items():
            dist = ((mean_lat - city_lat) ** 2 + (mean_lng - city_lng) ** 2) ** 0.5
            if dist < min_distance:
                min_distance = dist
                city = city_name
        
        # Get city center coordinates
        city_lat, city_lng = city_centers[city]
        
        # Calculate distance to center (Haversine formula)
        def haversine_distance(lat1, lng1, lat2, lng2):
            # Convert to radians
            lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r
        
        # Apply to all rows
        df_engineered['distance_to_center'] = df.apply(
            lambda row: haversine_distance(row['latitude'], row['longitude'], city_lat, city_lng),
            axis=1
        )
        
        # Create distance categories
        df_engineered['distance_category'] = pd.cut(
            df_engineered['distance_to_center'],
            bins=[0, 2, 5, 10, 100],
            labels=['City Center', 'Near Center', 'Suburbs', 'Remote']
        )
    
    # Neighborhood Features
    if 'Neighborhood Features' in selected_features:
        # Check which neighborhood column exists
        neighborhood_col = None
        if 'neighbourhood_cleansed' in df.columns:
            neighborhood_col = 'neighbourhood_cleansed'
        elif 'neighbourhood' in df.columns:
            neighborhood_col = 'neighbourhood'
        
        if neighborhood_col:
            # Get top neighborhoods
            top_neighborhoods = df[neighborhood_col].value_counts().nlargest(10).index
            
            # Create binary features for top neighborhoods
            for neighborhood in top_neighborhoods:
                col_name = f'neighborhood_{neighborhood.lower().replace(" ", "_")}'
                df_engineered[col_name] = df[neighborhood_col] == neighborhood
            
            # Create a feature for "other" neighborhoods
            df_engineered['neighborhood_other'] = ~df[neighborhood_col].isin(top_neighborhoods)
    
    # Availability Features
    if 'Availability Features' in selected_features and 'availability_365' in df.columns:
        # Create availability rate (percentage of the year available)
        df_engineered['availability_rate'] = df['availability_365'] / 365 * 100
        
        # Create availability categories
        df_engineered['availability_category'] = pd.cut(
            df['availability_365'],
            bins=[0, 30, 90, 180, 365],
            labels=['Rare', 'Occasional', 'Frequent', 'Always']
        )
        
        # Create binary feature for high availability
        df_engineered['high_availability'] = df['availability_365'] > 180
    
    # Text Length Features
    if 'Text Length Features' in selected_features:
        # Find text columns (name, description, summary, etc.)
        text_cols = [col for col in df.columns if pd.api.types.is_string_dtype(df[col])]
        
        for col in text_cols:
            # Only process columns that are likely to contain descriptions
            if col in ['name', 'description', 'summary', 'space', 'neighborhood_overview', 'notes', 'transit']:
                # Calculate text length
                df_engineered[f'{col}_length'] = df[col].fillna('').apply(len)
                
                # Create categories based on text length
                if df_engineered[f'{col}_length'].max() > 0:
                    quantiles = df_engineered[f'{col}_length'].quantile([0.33, 0.67])
                    
                    def categorize_length(length):
                        if length <= quantiles[0.33]:
                            return 'Short'
                        elif length <= quantiles[0.67]:
                            return 'Medium'
                        else:
                            return 'Long'
                    
                    df_engineered[f'{col}_length_category'] = df_engineered[f'{col}_length'].apply(categorize_length)
    
    # Amenities Count
    if 'Amenities Count' in selected_features and 'amenities' in df.columns:
        # Count number of amenities
        def count_amenities(amenities_str):
            if pd.isna(amenities_str):
                return 0
            
            # Clean up the amenities string and count items
            cleaned = re.sub(r'[\[\]\{\}"\'.,]', ' ', str(amenities_str))
            items = [item.strip() for item in cleaned.split(',') if item.strip()]
            return len(items)
        
        df_engineered['amenities_count'] = df['amenities'].apply(count_amenities)
        
        # Create categories based on amenities count
        amenities_quantiles = df_engineered['amenities_count'].quantile([0.33, 0.67]).values
        
        def categorize_amenities(count):
            if count <= amenities_quantiles[0]:
                return 'Basic'
            elif count <= amenities_quantiles[1]:
                return 'Standard'
            else:
                return 'Luxury'
        
        df_engineered['amenities_category'] = df_engineered['amenities_count'].apply(categorize_amenities)
        
        # Check for popular amenities
        popular_amenities = ['wifi', 'kitchen', 'washer', 'dryer', 'air conditioning', 'heating']
        
        for amenity in popular_amenities:
            df_engineered[f'has_{amenity.replace(" ", "_")}'] = df['amenities'].fillna('').str.lower().str.contains(amenity.lower())
    
    # Host Features
    if 'Host Features' in selected_features:
        # Check for host-related columns
        host_cols = [col for col in df.columns if 'host_' in col]
        
        if host_cols:
            # Host verification status
            if 'host_is_superhost' in df.columns:
                # Ensure it's boolean
                df_engineered['host_is_superhost'] = df['host_is_superhost'].astype(bool)
            
            # Host response time
            if 'host_response_time' in df.columns:
                # Create response time score (faster is higher)
                response_time_map = {
                    'within an hour': 4,
                    'within a few hours': 3,
                    'within a day': 2,
                    'a few days or more': 1,
                    np.nan: 0
                }
                df_engineered['host_response_score'] = df['host_response_time'].map(response_time_map)
            
            # Host experience (years since becoming host)
            if 'host_since' in df.columns:
                try:
                    # Convert to datetime if needed
                    if df['host_since'].dtype != 'datetime64[ns]':
                        df_engineered['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
                    
                    # Calculate years of experience
                    current_year = datetime.now().year
                    df_engineered['host_years_experience'] = current_year - df_engineered['host_since'].dt.year
                    
                    # Create experience categories
                    df_engineered['host_experience_category'] = pd.cut(
                        df_engineered['host_years_experience'],
                        bins=[-1, 1, 3, 5, 100],
                        labels=['New', 'Established', 'Experienced', 'Veteran']
                    )
                except:
                    pass
            
            # Host verification features
            if 'host_verifications' in df.columns:
                # Count verifications
                df_engineered['host_verification_count'] = df['host_verifications'].fillna('').apply(
                    lambda x: len(re.findall(r'[a-zA-Z_]+', str(x)))
                )
                
                # Check for specific verifications
                important_verifications = ['email', 'phone', 'government_id', 'identity_manual']
                for verification in important_verifications:
                    df_engineered[f'host_verified_{verification}'] = df['host_verifications'].fillna('').str.contains(verification)
                
                # Create a combined verification score
                df_engineered['host_verification_score'] = df_engineered[[f'host_verified_{v}' for v in important_verifications]].sum(axis=1)
    
    return df_engineered

def identify_patterns(df):
    """
    Identify patterns and trends in the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
        
    Returns:
    --------
    list
        List of identified patterns and trends
    """
    patterns = []
    
    # Price patterns
    if 'price' in df.columns:
        # Check price distribution
        price_mean = df['price'].mean()
        price_median = df['price'].median()
        price_ratio = price_mean / price_median if price_median > 0 else 0
        
        if price_ratio > 1.5:
            patterns.append(f"Price distribution is highly right-skewed (mean/median ratio: {price_ratio:.2f}), " 
                            f"indicating a small number of premium listings driving up the average price.")
        
        # Price by room type
        if 'room_type' in df.columns:
            room_prices = df.groupby('room_type')['price'].median().sort_values(ascending=False)
            if len(room_prices) > 1:
                highest_room = room_prices.index[0]
                lowest_room = room_prices.index[-1]
                price_diff = room_prices.iloc[0] / room_prices.iloc[-1] if room_prices.iloc[-1] > 0 else 0
                
                if price_diff > 0:
                    patterns.append(f"{highest_room} is {price_diff:.1f}x more expensive than {lowest_room} on average.")
        
        # Price by location (if available)
        if 'neighbourhood' in df.columns or 'neighbourhood_cleansed' in df.columns:
            location_col = 'neighbourhood_cleansed' if 'neighbourhood_cleansed' in df.columns else 'neighbourhood'
            
            # Filter to locations with at least 5 listings
            location_counts = df[location_col].value_counts()
            valid_locations = location_counts[location_counts >= 5].index
            
            if len(valid_locations) > 1:
                location_prices = df[df[location_col].isin(valid_locations)].groupby(location_col)['price'].median().sort_values(ascending=False)
                
                if len(location_prices) > 1:
                    highest_location = location_prices.index[0]
                    lowest_location = location_prices.index[-1]
                    price_diff = location_prices.iloc[0] / location_prices.iloc[-1] if location_prices.iloc[-1] > 0 else 0
                    
                    if price_diff > 0:
                        patterns.append(f"{highest_location} is {price_diff:.1f}x more expensive than {lowest_location} on average.")
    
    # Availability patterns
    if 'availability_365' in df.columns:
        # Overall availability
        avg_availability = df['availability_365'].mean()
        availability_rate = avg_availability / 365 * 100
        
        patterns.append(f"On average, listings are available {availability_rate:.1f}% of the year ({avg_availability:.1f} days).")
        
        # Availability by room type
        if 'room_type' in df.columns:
            room_availability = df.groupby('room_type')['availability_365'].mean().sort_values(ascending=False)
            
            if len(room_availability) > 1:
                most_available = room_availability.index[0]
                least_available = room_availability.index[-1]
                
                patterns.append(f"{most_available} listings have the highest availability ({room_availability.iloc[0]:.1f} days/year), " 
                                f"while {least_available} listings have the lowest ({room_availability.iloc[-1]:.1f} days/year).")
    
    # Review patterns
    review_cols = [col for col in df.columns if 'review' in col.lower() and 'score' in col.lower()]
    
    if review_cols:
        for col in review_cols[:1]:  # Just use the first review score column
            # Overall rating
            avg_rating = df[col].mean()
            max_possible = df[col].max()
            
            if max_possible > 0:
                rating_percent = avg_rating / max_possible * 100
                patterns.append(f"The average {col} is {avg_rating:.2f} out of {max_possible} ({rating_percent:.1f}%).")
            
            # Rating by room type
            if 'room_type' in df.columns:
                room_ratings = df.groupby('room_type')[col].mean().sort_values(ascending=False)
                
                if len(room_ratings) > 1:
                    highest_rated = room_ratings.index[0]
                    lowest_rated = room_ratings.index[-1]
                    
                    patterns.append(f"{highest_rated} listings have the highest {col} ({room_ratings.iloc[0]:.2f}), " 
                                    f"while {lowest_rated} listings have the lowest ({room_ratings.iloc[-1]:.2f}).")
    
    # Host patterns
    if 'host_is_superhost' in df.columns:
        superhost_rate = df['host_is_superhost'].mean() * 100
        patterns.append(f"{superhost_rate:.1f}% of hosts are Superhosts.")
        
        # Price by superhost status
        if 'price' in df.columns:
            superhost_price = df[df['host_is_superhost'] == True]['price'].median()
            regular_price = df[df['host_is_superhost'] == False]['price'].median()
            
            if superhost_price > 0 and regular_price > 0:
                price_diff = superhost_price / regular_price
                
                if price_diff > 1.1:
                    patterns.append(f"Superhost listings are {price_diff:.2f}x more expensive than regular host listings.")
                elif price_diff < 0.9:
                    patterns.append(f"Regular host listings are {1/price_diff:.2f}x more expensive than Superhost listings.")
    
    # Location patterns
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Check for geographical clusters
        from sklearn.cluster import KMeans
        
        # Use a small number of clusters for simplicity
        kmeans = KMeans(n_clusters=3, random_state=42)
        
        # Only use valid coordinates
        valid_coords = df[['latitude', 'longitude']].dropna()
        
        if len(valid_coords) > 3:
            kmeans.fit(valid_coords)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            # Count listings in each cluster
            df_copy = df.copy()
            df_copy['cluster'] = -1
            df_copy.loc[valid_coords.index, 'cluster'] = kmeans.labels_
            
            cluster_counts = df_copy['cluster'].value_counts()
            
            # Calculate percentage in largest cluster
            largest_cluster = cluster_counts.idxmax()
            largest_cluster_pct = cluster_counts[largest_cluster] / len(valid_coords) * 100
            
            if largest_cluster_pct > 50:
                patterns.append(f"{largest_cluster_pct:.1f}% of listings are concentrated in a single geographical cluster.")
    
    # Time patterns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if date_cols:
        # Try to find a listing date column
        listing_date_col = None
        for col in date_cols:
            if 'created' in col.lower() or 'listed' in col.lower() or 'first' in col.lower():
                listing_date_col = col
                break
        
        if listing_date_col:
            try:
                # Convert to datetime if needed
                if df[listing_date_col].dtype != 'datetime64[ns]':
                    date_series = pd.to_datetime(df[listing_date_col], errors='coerce')
                else:
                    date_series = df[listing_date_col]
                
                # Get year counts
                year_counts = date_series.dt.year.value_counts().sort_index()
                
                if len(year_counts) > 1:
                    # Calculate year-over-year growth
                    years = year_counts.index.tolist()
                    max_growth_year = None
                    max_growth_rate = 0
                    
                    for i in range(1, len(years)):
                        prev_year = years[i-1]
                        curr_year = years[i]
                        
                        prev_count = year_counts[prev_year]
                        curr_count = year_counts[curr_year]
                        
                        if prev_count > 0:
                            growth_rate = (curr_count - prev_count) / prev_count
                            
                            if growth_rate > max_growth_rate:
                                max_growth_rate = growth_rate
                                max_growth_year = curr_year
                    
                    if max_growth_year and max_growth_rate > 0.2:
                        patterns.append(f"The highest growth in new listings was in {max_growth_year} " 
                                        f"({max_growth_rate*100:.1f}% increase from the previous year).")
            except:
                pass
    
    return patterns

def calculate_derived_metrics(df):
    """
    Calculate derived metrics from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to calculate metrics for
        
    Returns:
    --------
    dict
        Dictionary of derived metrics
    """
    metrics = {}
    
    # Price metrics
    if 'price' in df.columns:
        metrics['avg_price'] = df['price'].mean()
        metrics['median_price'] = df['price'].median()
        metrics['price_range'] = df['price'].max() - df['price'].min()
        metrics['price_std'] = df['price'].std()
        metrics['price_per_accommodates'] = df['price'].sum() / df['accommodates'].sum() if 'accommodates' in df.columns else None
    
    # Occupancy metrics
    if 'availability_365' in df.columns:
        metrics['avg_availability'] = df['availability_365'].mean()
        metrics['avg_occupancy'] = 365 - metrics['avg_availability']
        metrics['occupancy_rate'] = metrics['avg_occupancy'] / 365 * 100
    
    # Review metrics
    review_cols = [col for col in df.columns if 'review' in col.lower() and 'score' in col.lower()]
    
    if review_cols:
        for col in review_cols:
            metrics[f'avg_{col}'] = df[col].mean()
            metrics[f'median_{col}'] = df[col].median()
    
    # Host metrics
    if 'host_is_superhost' in df.columns:
        metrics['superhost_rate'] = df['host_is_superhost'].mean() * 100
    
    if 'host_response_rate' in df.columns:
        try:
            # Convert percentage string to float
            response_rates = df['host_response_rate'].str.rstrip('%').astype('float') / 100
            metrics['avg_response_rate'] = response_rates.mean() * 100
        except:
            pass
    
    # Location metrics
    if 'neighbourhood' in df.columns or 'neighbourhood_cleansed' in df.columns:
        location_col = 'neighbourhood_cleansed' if 'neighbourhood_cleansed' in df.columns else 'neighbourhood'
        
        # Number of unique neighborhoods
        metrics['unique_neighborhoods'] = df[location_col].nunique()
        
        # Top 3 neighborhoods by listing count
        top_neighborhoods = df[location_col].value_counts().nlargest(3)
        metrics['top_neighborhoods'] = top_neighborhoods.to_dict()
    
    # Room type metrics
    if 'room_type' in df.columns:
        # Distribution of room types
        room_type_dist = df['room_type'].value_counts(normalize=True) * 100
        metrics['room_type_distribution'] = room_type_dist.to_dict()
    
    return metrics
