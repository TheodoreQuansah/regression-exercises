import seaborn as sns
import matplotlib.pyplot as plt

#_______________________________________________________________hypothesis test________________________________________________________________________

def hypothesis_test(p):
    alpha = 0.05
    if p < alpha:
        print('we reject the null, there is a relationship.')
    else:
        print('Fail to reject the null, there is no relationship.')



#_______________________________________________________________plotting variable pairs__________________________________________________________________

def plot_variable_pairs(df):

    # Select only the continuous variables from the DataFrame
    continuous_vars = df.select_dtypes(include=['float64', 'int64'])
    
    # Create a pair plot with regression lines
    sns.set(style="ticks")
    sns.pairplot(continuous_vars.sample(10_000), kind="reg", plot_kws={'line_kws':{'color':'red'}}, corner=True)
    
    # Display the plot
    plt.show()




#_____________________________________________________plot_categorical_and_continuous_vars_________________________________________________________________

def plot_categorical_and_continuous_vars(df, categorical_col, continuous_col):
    # Plot 1: Box plot to visualize the distribution of the continuous variable by category
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df.sample(10_000), x=categorical_col, y=continuous_col)
    plt.title(f'Distribution of {continuous_col} by {categorical_col}')
    plt.xticks(rotation=45)
    plt.xlabel(categorical_col)
    plt.ylabel(continuous_col)
    #plt.grid(True)
    plt.show()

    # Plot 2: Swarm plot to visualize the distribution of the continuous variable by category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df.sample(10_000), x=categorical_col, y=continuous_col)
    plt.title(f'Distribution of {continuous_col} by {categorical_col}')
    plt.xticks(rotation=45)
    plt.xlabel(categorical_col)
    plt.ylabel(continuous_col)
    #plt.grid(True)
    plt.show()

    # Plot 3: Bar plot to visualize the mean of the continuous variable by category
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df.sample(10_000), x=categorical_col, y=continuous_col)
    plt.title(f'{continuous_col} by {categorical_col}')
    plt.xticks(rotation=45)
    plt.xlabel(categorical_col)
    plt.ylabel(f'{continuous_col}')
    #plt.grid(True)
    plt.show()