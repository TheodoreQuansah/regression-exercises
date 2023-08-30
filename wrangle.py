import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from env import get_connection
from sklearn.model_selection import train_test_split

#___________________________________________________Aqcuiring the properties_2017 dataset from the zillow data base___________________________________________

def get_properties_2017():
    # Define the filename for the CSV file
    filename = 'properties_2017.csv'
    
    # Check if the CSV file already exists
    if os.path.isfile(filename):
        # If the file exists, read it into a DataFrame and return it
        return pd.read_csv(filename)
    else:
        # If the file doesn't exist, define an SQL query to retrieve data
        query = '''
            SELECT 
                bedroomcnt,
                bathroomcnt,
                calculatedfinishedsquarefeet,
                taxvaluedollarcnt,
                yearbuilt,
                taxamount,
                fips
            FROM
                properties_2017
            WHERE propertylandusetypeid = 261;
        '''
        
        # Get a connection URL (you may want to define the 'get_connection' function)
        url = get_connection('zillow')  # You'll need to define this function
        
        # Execute the SQL query and read the result into a DataFrame
        df = pd.read_sql(query, url)
        
        # Save the result to a CSV file
        df.to_csv(filename, index=False)

        # Return the DataFrame
        return df


#________________________________________________________________cleaning the properties_2017 dataset______________________________________________________________

def clean_and_convert():

    df = get_properties_2017()
    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'squarefeet',
        'taxamount' : 'tax_amount',
        'taxvaluedollarcnt': 'tax_value',
        'yearbuilt': 'year_built'
    })

    # Drop rows with any null values
    df = df.dropna()

    # Convert all columns to integers
    df = df.astype(int)

    return df


#________________________________________________________________splitting the dataset______________________________________________________________

def train_val_test(df):
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    # Return the three datasets
    return train, val, test


#________________________________________________________________creating a subplot______________________________________________________________

def compare_data(scaled_col, x_lim, df, original='tax_value'):
    # Create a figure with two side-by-side subplots for comparison
    plt.figure(figsize=(11, 7))
    
    # Left Subplot: Original Data Histogram
    plt.subplot(1, 2, 1)
    
    # Create a histogram of the original data (original column)
    sns.histplot(data=df, x=original, bins=20)
    
    # Set x-axis limits for the original data histogram
    plt.xlim(0, 90_000_000)
    
    # Set y-axis limits for the original data histogram
    plt.ylim(0, 100)
    
    # Right Subplot: Scaled Data Histogram
    plt.subplot(1, 2, 2)
    
    # Create a histogram of the scaled data (scaled_col)
    sns.histplot(data=df, x=scaled_col, bins=20)
    
    # Set x-axis limits for the scaled data histogram
    plt.xlim(0, x_lim)
    
    # Set y-axis limits for the scaled data histogram
    plt.ylim(0, 100)
    
    # Display the side-by-side histograms
    plt.show()
