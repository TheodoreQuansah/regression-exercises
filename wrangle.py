import os
import pandas as pd

from env import get_connection

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

def clean_and_convert(df):
    # Rename columns
    df = df.rename(columns={
        'bedroomcnt': 'bedrooms',
        'bathroomcnt': 'bathrooms',
        'calculatedfinishedsquarefeet': 'Squarefeet',
        'taxvaluedollarcnt': 'tax_value',
        'yearbuilt': 'year_built'
    })

    # Drop rows with any null values
    df = df.dropna()

    # Convert all columns to integers
    df = df.astype(int)

    return df