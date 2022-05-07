import pdb
import pandas as pd
import numpy as np
from pyspark import SparkContext, SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
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

    # Use the last 10% of days as test data, so that we're not training on future data.
    cutoff = df['timestamp'].quantile(0.9)
    train_df = df[df['timestamp'] < cutoff]
    test_df = df[df['timestamp'] >= cutoff]

    # Get rid of the timestamp column, which is no longer needed
    train_df = train_df[['userId', 'movieId', 'rating']]
    train_df = train_df.rename(columns={'userId': 'user', 'movieId': 'item'})
    test_df = test_df[['userId', 'movieId', 'rating']]
    test_df = test_df.rename(columns={'userId': 'user', 'movieId': 'item'})

    return train_df, test_df


def train_recommender(sql_context, train_df):
    """
    Given a SQL context and train and test pandas dataframes, perform ALS on the data and generate recommendations using
    Pyspark.
    """

    rank = 10
    n_iter = 10

    train_df = sql_context.createDataFrame(train_df)
    als = ALS(rank=rank, seed=SEED, maxIter=n_iter)
    model = als.fit(train_df)

    return model


def evaluate(sql_context, model, test_df):
    """Evaluate the performance of the model on held-out data."""

    test_df = sql_context.createDataFrame(test_df)
    evaluator = RegressionEvaluator()
    predictions = model.transform(test_df.select(['user', 'item']))
    predictions = predictions.na.drop()  # remove rows with NaN. These are cold start users.
    labels_and_preds = predictions.join(test_df, on=['user', 'item'])\
        .withColumnRenamed('rating', 'label')

    mae = evaluator.evaluate(labels_and_preds, {evaluator.metricName: 'mae'})
    r2 = evaluator.evaluate(labels_and_preds, {evaluator.metricName: 'r2'})
    rmse = evaluator.evaluate(labels_and_preds, {evaluator.metricName: 'rmse'})
    print(f'MAE: {mae}')
    print(f'R2: {r2}')
    print(f'RMSE: {rmse}')


if __name__ == '__main__':

    sc = SparkContext('local', 'recommendation app')
    sql_context = SQLContext(sc)
    print(sc.version)

    ratings_fn = './data/ratings.csv'
    train_df, test_df = load_data(ratings_fn)
    model = train_recommender(sql_context, train_df)
    evaluate(sql_context, model, test_df)
