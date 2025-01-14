import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def bin_to_num(data):
    binnedInc = []
    for i in data["binnedInc"]:
        # remove the parentheses and brackets
        i =i.strip("()[]")
        #  split the string into a list after splitting by coma
        i = i.split(",")
        # convert the list to a tuple
        i = tuple(i)
        # convert individual elements to float
        i = tuple(map(float, i))
        # convert the tuple to a list
        i = list(i)
        # append the list to the biinedIn list
        binnedInc.append(i)
    data["binnedInc"] = binnedInc

    # make a new column lower and upper bound
    data["lower_bound"] = [i[0] for i in data["binnedInc"]]
    data["upper_bound"] = [i[1] for i in data["binnedInc"]]

    # and also median point 
    data["median"] = (data["lower_bound"] + data["upper_bound"])/2

    # drop the binnedInc column
    data.drop("binnedInc", axis = 1, inplace= True)
    return data


def cat_to_col(data):
    # make a new column by splitting the 
    data["county"] = [i.split(",")[0] for i in data["Geography"]]
    data["state"] = [i.split(",")[1] for i in data["Geography"]]
    # drop the geography column
    data.drop("Geography", axis=1, inplace=True)
    return data


def one_hot_encoding(X):
    # select categorical columns
    categorical_columns = X.select_dtypes(include=["object"]).columns
    # one hot encode categroical columns
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    one_hot_encoded = one_hot_encoder.fit_transform(X[categorical_columns])
    # convert the one hot encoded array to a dataframe
    one_hot_encoded = pd.DataFrame(
        one_hot_encoded, columns= one_hot_encoder.get_feature_names_out(categorical_columns)
    )
    # drop the categorical columns from the original dataframe
    X = X.drop(categorical_columns, axis=1)
    # concatenate the one hot encoded dataframe to the original dataframe
    X = pd.concat([X,one_hot_encoded], axis =1)
    return X