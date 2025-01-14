from sklearn.model_selection import train_test_split as tts


def find_constant_columns(dataframe):
    """
    This function takes in a dataframe and returns the columns that contain a single value.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed

    Returns:
    list: A list of columns that contain a single value
    """

    constant_columns = []
    for column in dataframe.columns:
        # Get unique values in the column
        unique_values = dataframe[column].unique()
        #check if the column contains only one unique value
        if len(unique_values) ==1:
            constant_columns.append(column)
        
    return constant_columns


def drop_and_fill(dataframe):
    # Get the columns with more than 50% missing values
    cols_to_drop = dataframe.columns[dataframe.isnull().mean() > 0.5]
    # Drop the columns 
    dataframe = dataframe.drop(cols_to_drop, axis =1)
    # Fill the remaining missing values with mean of the column
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe


def find_columns_with_few_values(dataframe, threshold):
    """
    This function takes in a dataframe and a threshold value

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to be analyzed
    threshold (int): The minmum number of uniques values

    Returns:
    list: A list of columns that have less than the threshold values
    """

    few_values_columns = []
    for column in dataframe.columns:
        # Get the number of unique values in the column
        unique_values_count = len(dataframe[column].unique())
        # Check if the column has less than the threshold
        if unique_values_count < threshold:
            few_values_columns.append(column)
    return few_values_columns


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.
    
    Parameters:
    - df: pandas DataFrame containing the data
    - target_column: the name of the column to be predicted (dependent variable)
    - test_size: proportion of the dataset to include in the test split (default: 0.2)
    - random_state: seed for reproducibility (default: 42)
    
    Returns:
    - x_train, x_test: independent variables for training and testing
    - y_train, y_test: dependent variable for training and testing
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    x_train, x_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test