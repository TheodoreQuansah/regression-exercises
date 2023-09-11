from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def eval_model(y_actual, y_hat):
    
    return sqrt(mean_squared_error(y_actual, y_hat))


def train_model(model, X_train, y_train, X_val, y_val):
    
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    
    train_rmse = eval_model(y_train, train_preds)
    
    val_preds = model.predict(X_val)
    
    val_rmse = eval_model(y_val, val_preds)
    
    print(f'The train RMSE is {train_rmse}.')
    print(f'The validate RMSE is {val_rmse}.')
    
    return model




def pearson_correlation(x, y, alpha=0.05):
    """
    Calculate Pearson's correlation coefficient (r) between two arrays or lists of data and perform hypothesis testing.

    Parameters:
    x (array-like): The first data array.
    y (array-like): The second data array.
    alpha (float): The significance level for hypothesis testing (default is 0.05).

    Returns:
    str: A result message based on hypothesis testing.
    float: Pearson's correlation coefficient (r) between x and y.
    float: p-value associated with the correlation coefficient.

    Note:
    - The function returns both the correlation coefficient (r) and the p-value.
    - The p-value indicates the statistical significance of the correlation.
    - If p-value is less than alpha, you can reject the null hypothesis of no correlation.
    """
    correlation_coefficient, p_value = stats.pearsonr(x, y)
    
    if p_value < alpha:
        result = "We reject the null hypothesis. There appears to be a relationship."
    else:
        result = "We fail to reject the null hypothesis."
    
    return result, (f'r = {correlation_coefficient}.'), (f'p = {p_value}.')





def perform_anova(df, alpha=0.05):
    """
    Perform ANOVA on tax_value by FIPS regions.

    Parameters:
    - df: DataFrame containing 'tax_value' and 'fips' columns.
    - alpha: Significance level for hypothesis testing (default is 0.05).

    Returns:
    - A string indicating the result of the ANOVA test.
    """
    # Perform ANOVA
    anova_result = stats.f_oneway(
        df[df['fips'] == 6037]['tax_value'],
        df[df['fips'] == 6059]['tax_value'],
        df[df['fips'] == 6111]['tax_value']
    )

    if anova_result.pvalue < alpha:
        result = "We reject the null hypothesis. There is a significant difference among FIPS regions."
    else:
        result = "We fail to reject the null hypothesis."

    return result, anova_result.pvalue



def pearson_correlation_test(data1, data2):
    """
    Perform a Pearson correlation test between two datasets.

    Parameters:
    data1 (pd.Series): First dataset for correlation analysis.
    data2 (pd.Series): Second dataset for correlation analysis.

    Returns:
    correlation_coefficient (float): Pearson correlation coefficient.
    p_value (float): Two-tailed p-value.
    """

    alpha = 0.05
    correlation_coefficient, p_value = stats.pearsonr(data1, data2)

    if p_value < alpha:
        result = "We reject the null hypothesis. There is a correlation between tax value and number of bedrooms."
    else:
        result = "We fail to reject the null hypothesis."
    
    return result, correlation_coefficient, p_value



def create_scatter_plot_sf(data, x_column, y_column):
    """
    Create a scatter plot.

    Parameters:
    - x: Data for the x-axis.
    - y: Data for the y-axis.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title for the plot.

    Returns:
    - None (displays the plot).
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title('Scatter Plot of Square Feet vs. Tax Value')
    plt.xlabel('Square Feet')
    plt.ylabel('Tax Value')
    plt.grid(True)
    plt.show()

# Example usage:
# create_scatter_plot(df['squarefeet'], df['tax_value'], 'Square Feet', 'Tax Value', 'Scatter Plot of Square Feet vs. Tax Value')





def plot_tax_value_by_fips(data):
    """
    Create a barplot of tax_value by FIPS region.

    Parameters:
    - data: DataFrame containing the tax_value and fips columns.

    Returns:
    - None (displays the plot).
    """
    sns.barplot(data=data, x='fips', y='tax_value')
    plt.title("Bar Plot of Tax Value by FIPS Region")
    plt.xlabel("FIPS Region")
    plt.ylabel("Tax Value")
    plt.show()





def scatter_plot_bedrooms_vs_tax_value(bedrooms, tax_value):
    """
    Create a scatter plot to visualize the correlation between bedrooms and tax value.

    Parameters:
    bedrooms (pd.Series): Column containing the number of bedrooms.
    tax_value (pd.Series): Column containing tax values.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.bar(bedrooms, tax_value)
    plt.title('Correlation Between Bedrooms and Tax Value')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Tax Value')
    plt.grid(True)
    plt.show()







def create_barplot(data, x_column, y_column):
    """
    Create a bar plot to visualize the relationship between two columns.

    Parameters:
    data (DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column for the x-axis.
    y_column (str): The name of the column for the y-axis.
    title (str): The title of the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x_column, y=y_column)
    plt.title('Bathrooms vs. Tax Value')
    plt.xlabel('Bathrooms (Categories)')
    plt.ylabel('Tax Value')
    plt.grid(True)
    plt.show()






def create_scatter_plot(data, x_column, y_column):
    """
    Create a scatter plot to visualize the relationship between two columns.

    Parameters:
    data (DataFrame): The DataFrame containing the data.
    x_column (str): The name of the column to be plotted on the x-axis.
    y_column (str): The name of the column to be plotted on the y-axis.
    title (str): The title of the plot.
    x_label (str): The label for the x-axis.
    y_label (str): The label for the y-axis.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_column], data[y_column], alpha=0.5)
    plt.title('Year Built vs. Tax Value')
    plt.xlabel('Year Built')
    plt.ylabel('Tax Value')
    plt.grid(True)
    plt.show()

