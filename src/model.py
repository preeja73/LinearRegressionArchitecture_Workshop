from sklearn.linear_model import LinearRegression


def train(X, y):
    """
    Train a Linear Regression model on the provided data.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix (input variables) for training
    y : array-like
        Target vector (output variable) for training
        
    Returns:
    --------
    LinearRegression
        Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model
