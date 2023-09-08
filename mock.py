import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from env import get_connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

#___________________________________________________Aqcuiring the properties_2017 dataset from the zillow data base___________________________________________

def get_properties_2017():
    # Define the filename for the CSV file
    filename = 'properties_2017mock3.csv'
    
    # Check if the CSV file already exists
    if os.path.isfile(filename):
        # If the file exists, read it into a DataFrame and return it
        return pd.read_csv(filename)
    else:
        # If the file doesn't exist, define an SQL query to retrieve data
        query = '''
                SELECT DISTINCT *
                FROM properties_2017
                LEFT JOIN predictions_2017 ON predictions_2017.parcelid = properties_2017.parcelid
                WHERE EXTRACT(YEAR FROM predictions_2017.transactiondate) = 2017
                AND propertylandusetypeid = 261;
                '''
        
        # Get a connection URL (you may want to define the 'get_connection' function)
        url = get_connection('zillow')  # You'll need to define this function
        
        # Execute the SQL query and read the result into a DataFrame
        df = pd.read_sql(query, url)
        
        # Save the result to a CSV file
        df.to_csv('filenames', index=False)

        # Return the DataFrame
        return df


#________________________________________________________________cleaning the properties_2017 dataset______________________________________________________________

def clean_and_converts():

    # Replace this line with how you obtain your DataFrame
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

    bed_edges = [0, 2, 3, 4, 6, 8, 11, 25]
    bath_edges = [0, 1, 2, 3, 4, 7, 15, 32]
    sf_edges = [1, 500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500, 5_000]
    dec_edges = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

    # Apply cut and assign the results to respective columns
    df['bedrooms_bin'] = pd.cut(df['bedrooms'], bins=bed_edges, right=False)
    df['bathrooms_bin'] = pd.cut(df['bathrooms'], bins=bath_edges, right=False)
    df['squarefeet_bin'] = pd.cut(df['squarefeet'], bins=sf_edges, right=False)
    df['decades'] = pd.cut(df['year_built'], bins=dec_edges, right=False)
    
    df['bedrooms_bin'] = df['bedrooms_bin'].apply(lambda x: x.right)
    df['bathrooms_bin'] = df['bathrooms_bin'].apply(lambda x: x.right)
    df['squarefeet_bin'] = df['squarefeet_bin'].apply(lambda x: x.right)
    df['decades'] = df['decades'].apply(lambda x: x.right)

    #Drop rows with any null values
    #df = df.dropna()

    #Convert all columns to integers
    #df = df.astype(int)

    return df


#________________________________________________________________splitting the dataset______________________________________________________________

def train_val_test(df):
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    # Return the three datasets
    return train, val, test
 

#________________________________________________________________creating a scaled subplot______________________________________________________________

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



#_____________________________________________________creating a quantile transformer subplot______________________________________________________________

# Create a QuantileTransformer with 'normal' output distribution
def quantiletransformer(train, output_distribution=None):
    # Check if output_distribution is provided, if not, set it to the default value
    if output_distribution is None:
        output_distribution = 'uniform'  # You can choose the default you prefer
    
    qt = QuantileTransformer(output_distribution=output_distribution, random_state=42)
    
    # Fit and transform your data using the QuantileTransformer
    train['tax_value_qt'] = qt.fit_transform(train[['tax_value']])
    
    # Create subplots to visualize the data before and after scaling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot the original data
    sns.histplot(data=train, x=train.tax_value, bins=50, ax=ax1)
    ax1.set_title('Original data')
    
    # Plot the scaled data
    sns.histplot(data=train, x=train.tax_value_qt, bins=50, ax=ax2)
    ax2.set_title('Scaled Data (QuantileTransformer)')

    plt.tight_layout()
    plt.show()


#_____________________________________________________creating a mms scaled function______________________________________________________________

def scaled_data(train, val, test, to_scale, scaler_type='standard'):
    
    # Initialize the selected scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', 'robust', 'quantile'.")

    # Fit the scaler on the training data and transform all sets
    train[to_scale] = scaler.fit_transform(train[to_scale])
    val[to_scale] = scaler.transform(val[to_scale])
    test[to_scale] = scaler.transform(test[to_scale])

    return train, val, test




#_____________________________________________________Splitting X and y______________________________________________________________

def xy_split(df):
    
    return df.drop(columns=['tax_value']), df.tax_value


def scale_data(train, val, test, to_scale):
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = val.copy()
    test_scaled = test.copy()

    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(val[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled
