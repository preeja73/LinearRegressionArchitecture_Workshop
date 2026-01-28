from sklearn.model_selection import train_test_split


def split(X, y, test_size):
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (input variables)
    y : array-like
        Target vector (output variable)
    test_size : float
        Proportion of the dataset to include in the test split (e.g., 0.2 for 20%)
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test - Split datasets
    """
    return train_test_split(X, y, test_size=test_size)
