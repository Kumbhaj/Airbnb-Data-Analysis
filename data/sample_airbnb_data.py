import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate a sample Airbnb dataset
def generate_sample_airbnb_data(num_samples=1000):
    """
    Generate a sample Airbnb dataset with realistic features
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample Airbnb dataset
    """
    # Define room types and their probabilities
    room_types = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    room_probs = [0.6, 0.3, 0.05, 0.05]
    
    # Define neighborhoods
    neighborhoods = [
        'Downtown', 'Midtown', 'Uptown', 'Westside', 'Eastside', 
        'Northside', 'Southside', 'Old Town', 'New District', 'Central',
        'Riverside', 'Beachfront', 'University Area', 'Arts District', 'Financial District'
    ]
    
    # Define amenities
    all_amenities = [
        'Wifi', 'Kitchen', 'Heating', 'Air conditioning', 'Washer', 'Dryer', 
        'TV', 'Iron', 'Essentials', 'Hangers', 'Hair dryer', 'Laptop-friendly workspace',
        'Hot water', 'Cooking basics', 'Elevator', 'Free parking', 'Pool', 'Gym', 
        'Hot tub', 'Breakfast', 'Indoor fireplace', 'Smoke detector', 'Carbon monoxide detector',
        'First aid kit', 'Fire extinguisher', 'Lock on bedroom door', 'Private entrance'
    ]
    
    # Generate listing IDs
    listing_ids = np.arange(1, num_samples + 1)
    
    # Generate host information
    host_ids = np.random.randint(1, num_samples // 5, size=num_samples)
    host_since = [
        datetime(2015, 1, 1) + timedelta(days=np.random.randint(0, 365 * 8)) 
        for _ in range(num_samples)
    ]
    host_response_times = np.random.choice(
        ['within an hour', 'within a few hours', 'within a day', 'a few days or more'], 
        size=num_samples,
        p=[0.4, 0.3, 0.2, 0.1]
    )
    host_response_rates = np.random.uniform(0.7, 1.0, size=num_samples)
    host_is_superhost = np.random.choice([True, False], size=num_samples, p=[0.2, 0.8])
    host_total_listings = np.random.exponential(2, size=num_samples).astype(int) + 1
    
    # Generate verifications with different probabilities
    verifications = ['email', 'phone', 'identity_manual', 'government_id', 'selfie', 'work_email']
    host_verifications = []
    for _ in range(num_samples):
        verification_count = np.random.randint(1, len(verifications) + 1)
        selected = np.random.choice(verifications, size=verification_count, replace=False)
        host_verifications.append(str(list(selected)))
    
    # Generate location information
    latitudes = np.random.uniform(40.70, 40.80, size=num_samples)  # New York-like coordinates
    longitudes = np.random.uniform(-74.02, -73.92, size=num_samples)
    
    # Generate room information
    room_type = np.random.choice(room_types, size=num_samples, p=room_probs)
    accommodates = np.random.randint(1, 11, size=num_samples)
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], size=num_samples, p=[0.3, 0.15, 0.25, 0.1, 0.1, 0.05, 0.05])
    bedrooms = np.random.randint(1, 6, size=num_samples)
    beds = np.random.randint(1, 8, size=num_samples)
    
    # Function to generate amenities string
    def generate_amenities():
        count = np.random.randint(5, 20)
        selected = random.sample(all_amenities, min(count, len(all_amenities)))
        return str(selected)
    
    # Generate amenities
    amenities = [generate_amenities() for _ in range(num_samples)]
    
    # Generate price information based on room type and location
    base_prices = {
        'Entire home/apt': 150,
        'Private room': 80,
        'Shared room': 50,
        'Hotel room': 120
    }
    
    prices = []
    for i in range(num_samples):
        # Base price from room type
        base = base_prices[room_type[i]]
        
        # Adjust for number of accommodates
        base *= (1 + 0.1 * (accommodates[i] - 1))
        
        # Add some random variation
        noise = np.random.uniform(0.7, 1.3)
        price = base * noise
        
        # Round to nearest 5
        price = round(price / 5) * 5
        prices.append(price)
    
    # Generate review information
    first_review = [
        host_since[i] + timedelta(days=np.random.randint(30, 365)) 
        for i in range(num_samples)
    ]
    last_review = [
        first_review[i] + timedelta(days=np.random.randint(0, 365 * 3)) 
        for i in range(num_samples)
    ]
    
    # Some listings have no reviews
    has_reviews = np.random.choice([True, False], size=num_samples, p=[0.85, 0.15])
    for i in range(num_samples):
        if not has_reviews[i]:
            first_review[i] = None
            last_review[i] = None
    
    # Generate review scores
    def generate_review_score(base=4.7, std=0.3):
        score = np.random.normal(base, std)
        return min(max(score, 1), 5)  # Clamp between 1 and 5
    
    review_scores_rating = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_accuracy = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_cleanliness = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_checkin = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_communication = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_location = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    review_scores_value = [generate_review_score() if has_reviews[i] else None for i in range(num_samples)]
    
    # Number of reviews with some correlation to superhost status
    number_of_reviews = []
    for i in range(num_samples):
        if not has_reviews[i]:
            number_of_reviews.append(0)
        else:
            base_reviews = 5
            if host_is_superhost[i]:
                base_reviews = 20
            num_reviews = int(np.random.exponential(base_reviews)) + 1
            number_of_reviews.append(num_reviews)
    
    # Generate availability information
    availability_30 = np.random.randint(0, 31, size=num_samples)
    availability_60 = availability_30 + np.random.randint(0, 31, size=num_samples)
    availability_90 = availability_60 + np.random.randint(0, 31, size=num_samples)
    availability_365 = np.random.randint(0, 366, size=num_samples)
    
    # Generate neighborhood information
    neighbourhood = np.random.choice(neighborhoods, size=num_samples)
    neighbourhood_cleansed = neighbourhood.copy()  # In a real dataset these might differ
    
    # Generate text fields
    def generate_text(word_count_range=(5, 15)):
        word_count = np.random.randint(word_count_range[0], word_count_range[1] + 1)
        words = ['Lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 
                 'elit', 'sed', 'do', 'eiusmod', 'tempor', 'incididunt', 'ut', 'labore', 
                 'et', 'dolore', 'magna', 'aliqua', 'Ut', 'enim', 'ad', 'minim', 'veniam',
                 'quis', 'nostrud', 'exercitation', 'ullamco', 'laboris', 'nisi', 'ut',
                 'aliquip', 'ex', 'ea', 'commodo', 'consequat']
        return ' '.join(np.random.choice(words, size=word_count))
    
    names = [f"Cozy {room_type[i]} in {neighbourhood[i]}" for i in range(num_samples)]
    descriptions = [generate_text((50, 200)) for _ in range(num_samples)]
    neighborhood_overview = [generate_text((20, 100)) for _ in range(num_samples)]
    
    # Create a dictionary for the DataFrame
    data = {
        'id': listing_ids,
        'name': names,
        'host_id': host_ids,
        'host_since': host_since,
        'host_response_time': host_response_times,
        'host_response_rate': host_response_rates,
        'host_is_superhost': host_is_superhost,
        'host_total_listings_count': host_total_listings,
        'host_verifications': host_verifications,
        'neighbourhood': neighbourhood,
        'neighbourhood_cleansed': neighbourhood_cleansed,
        'latitude': latitudes,
        'longitude': longitudes,
        'room_type': room_type,
        'accommodates': accommodates,
        'bathrooms': bathrooms,
        'bedrooms': bedrooms,
        'beds': beds,
        'amenities': amenities,
        'price': prices,
        'number_of_reviews': number_of_reviews,
        'first_review': first_review,
        'last_review': last_review,
        'review_scores_rating': review_scores_rating,
        'review_scores_accuracy': review_scores_accuracy,
        'review_scores_cleanliness': review_scores_cleanliness,
        'review_scores_checkin': review_scores_checkin,
        'review_scores_communication': review_scores_communication,
        'review_scores_location': review_scores_location,
        'review_scores_value': review_scores_value,
        'availability_30': availability_30,
        'availability_60': availability_60,
        'availability_90': availability_90,
        'availability_365': availability_365,
        'description': descriptions,
        'neighborhood_overview': neighborhood_overview
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values to make the dataset more realistic
    # For each column, randomly set some values to NaN
    for col in df.columns:
        # Skip id column
        if col == 'id':
            continue
        
        # Different missing rates for different columns
        if col in ['host_response_time', 'host_response_rate']:
            missing_rate = 0.1
        elif col in ['bathrooms', 'bedrooms', 'beds']:
            missing_rate = 0.05
        elif 'review' in col:
            missing_rate = 0.2
        else:
            missing_rate = 0.03
        
        # Create a mask for missing values
        mask = np.random.random(len(df)) < missing_rate
        df.loc[mask, col] = np.nan
    
    return df

# Generate and save the sample dataset
if __name__ == "__main__":
    df = generate_sample_airbnb_data(1000)
    df.to_csv('data/sample_airbnb_listings.csv', index=False)
    print(f"Sample dataset with {len(df)} records created at 'data/sample_airbnb_listings.csv'")