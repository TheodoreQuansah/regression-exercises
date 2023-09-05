import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#_______________________________________________________________creates a residual plot________________________________________________________________________

def plot_residuals(y, yhat):
    # Calculate the residuals (differences between observed and predicted values)
    y_hat_resids = yhat - y

    # Create a scatter plot of observed values against residuals
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y, y=y_hat_resids)
    plt.axhline(y=0, color='r', linestyle='--')
        
    # Adding labels and title
    plt.xlabel("Observed Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
        
    # Showing the plot
    plt.grid(True)
    plt.show()




#_______________________________________________________________model regression errors________________________________________________________________________

def regression_errors(y, yhat):
    
    # Calculate the residuals
    residuals = y - yhat

    # Calculate the Sum of Squared Errors (SSE)
    sse = np.sum(residuals**2)

    # Calculate the Explained Sum of Squares (ESS)
    ess = np.sum((yhat - np.mean(y))**2)

    # Calculate the Total Sum of Squares (TSS)
    tss = np.sum((y - np.mean(y))**2)

    # Calculate the Mean Squared Error (MSE)
    mse = sse / len(y)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Create a dictionary to store the results
    error_metrics = {
        'SSE': sse,
        'ESS': ess,
        'TSS': tss,
        'MSE': mse,
        'RMSE': rmse
    }

    return error_metrics




#_______________________________________________________________baseline mean errors________________________________________________________________________

def baseline_mean_errors(y):

    # calculate y_baseline
    y_baseline = y.median()
    
    # Calculate the residuals
    residuals = y - y_baseline

    # Calculate the Sum of Squared Errors (SSE)
    sse = np.sum(residuals**2)

    # Calculate the Explained Sum of Squares (ESS)
    ess = np.sum((y_baseline - np.mean(y))**2)

    # Calculate the Total Sum of Squares (TSS)
    tss = np.sum((y - np.mean(y))**2)

    # Calculate the Mean Squared Error (MSE)
    mse = sse / len(y)

    # Calculate the Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Create a dictionary to store the results
    error_metrics = {
        'SSE': sse,
        'ESS': ess,
        'TSS': tss,
        'MSE': mse,
        'RMSE': rmse
    }

    return error_metrics




#_______________________________________________________________better than baseline________________________________________________________________________

def better_than_baseline(y, yhat):

    # Calculate y_baseline
    y_baseline = y.median()
    
    # Calculate the residuals for the baseline
    b_residuals = y - y_baseline

    # Calculate the sse for the baseline
    sse_baseline = np.sum(b_residuals**2)

    # Calculate the residuals for the model
    residuals = y - yhat

    # Calculate the sse for the model
    sse_hat = np.sum(residuals**2)

    if sse_hat < sse_baseline:
        performance = "True"
    else:
        performance = "False"
    print(performance) 



