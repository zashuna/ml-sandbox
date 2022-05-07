import pdb
import pandas as pd
import numpy as np
import pyspark
from sklearn.model_selection import train_test_split
SEED = 2022


def load_data(filename):
    """
    Load the data, perform EDA, and split it into training and test sets.
    """

    df = pd.read_csv(filename)
    # Do some simple EDA
    df.info()
    df.describe()
    df['rating'].value_counts()

    train, test = train_test_split(df, test_size=0.1, random_state=SEED)
    return train, test


def generate_recommendations(train, test):
    """
    Given train and test pandas dataframes, perform ALS on the data and generate recommendations using Pyspark.
    """

    pass
