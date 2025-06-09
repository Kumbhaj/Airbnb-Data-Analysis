# Airbnb Data Analysis Project

A comprehensive data analysis tool for Airbnb listings data with interactive visualizations and in-depth statistical analysis.

In this particular instance I have used the dataset from kaggle (mentioned below), to test the features and all

If you want to test this Project you guys can try it on streamlit - https://airbnbdata.streamlit.app

Sample data you can download from kaggle - https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata?select=Airbnb_Open_Data.csv


<img width="1440" alt="Screenshot 2025-05-25 at 7 27 07 AM" src="https://github.com/user-attachments/assets/cf4bfbff-4ea2-4e3f-ae56-a08ebb260b42" />
<img width="1440" alt="Screenshot 2025-05-25 at 7 27 50 AM" src="https://github.com/user-attachments/assets/3fc9ddd4-20d3-4ac4-9d90-3bef790862a7" />
<img width="1440" alt="Screenshot 2025-05-25 at 7 28 17 AM" src="https://github.com/user-attachments/assets/f71c9e82-ebc1-420f-b6ec-bd213e36ef3c" />

## Features

This application performs detailed analysis on Airbnb listing data, covering:

### Data Cleaning and Preprocessing
- Automated detection and handling of missing values
- Duplicate record identification and removal
- Data type standardization and consistency checks
- Data integrity validation

### Feature Engineering
- Price categorization (Budget, Mid-range, Luxury)
- Location-based features including distance to center
- Host experience and verification metrics
- Amenities analysis and categorization
- Availability patterns identification

### Statistical Analysis
- Comprehensive summary statistics with interpretations
- Correlation analysis between key metrics
- Distribution analysis of numerical attributes
- Identification of key statistical relationships

### Pattern Identification
- Price variation across neighborhoods and property types
- Seasonal trends and patterns
- Host behavior and listing characteristics correlations
- Market positioning analysis

### Outlier Analysis
- Multiple outlier detection methods (Z-Score, IQR)
- Visualization of outlier distribution
- Impact analysis of outliers on overall patterns
- Outlier handling strategies (removal, capping, transformation)

### Visualization Dashboard
- Interactive price heatmaps by location
- Property distribution maps
- Correlation matrices
- Amenities word clouds
- Availability calendars

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries: streamlit, pandas, numpy, matplotlib, seaborn, plotly, scipy, wordcloud, statsmodels

### Installation

1. Clone this repository or download the source code
   ```
   git clone https://github.com/mehul-tandon/Airbnb-data-analysis.git
   cd airbnb-data-analysis
   ```

2. Install required packages
   ```
   pip install streamlit pandas numpy matplotlib seaborn plotly scipy wordcloud statsmodels
   ```

3. Run the application
   ```
   streamlit run app.py
   ```

   If port 5000 is already in use, you can specify a different port:
   ```
   streamlit run app.py --server.port 8501
   ```

4. The application will open in your default web browser

### Using the Application

1. Use the example dataset by checking the "Use example dataset" option in the sidebar
2. Or upload your own Airbnb dataset in CSV format
3. Navigate through the different analysis tabs to explore your data
4. Interact with the visualizations to gain insights
5. Apply various data processing techniques to clean and transform your data

## Project Structure

- `app.py`: Main Streamlit application file
- `data_processing.py`: Functions for data loading, cleaning, and preprocessing
- `analysis.py`: Statistical analysis and pattern identification functions
- `visualization.py`: Data visualization functions
- `utils.py`: Utility functions for formatting and display
- `data/`: Directory containing sample data and data generation scripts

## Project Focus Areas

This project focuses on several key data analysis techniques:

- Data cleaning and handling missing values
- Feature selection and engineering
- Ensuring data integrity and consistency
- Summary statistics and insights
- Identifying patterns, trends, and anomalies
- Handling outliers and data transformations
- Visual representation of key findings

## Contributing

This is an academic project but suggestions and improvements are welcome. Please feel free to fork the repository and submit pull requests.

## License

This project is available for educational and personal use.

## Acknowledgments

- Inside Airbnb for inspiring the data structure
- Streamlit for the interactive web application framework
- Python data science community for the excellent libraries
- Streamlit for the deployment of the project
- Kaggle for the Airbnb Open Dataset (https://www.kaggle.com/datasets/arianazmoudeh/airbnbopendata?select=Airbnb_Open_Data.csv)
