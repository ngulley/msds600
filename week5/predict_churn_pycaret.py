import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath, index_col='customerID')
    return df


def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    model = load_model('pycaret_churn_model')
    predictions = predict_model(model, data=df)
    predictions.rename({'Label': 'prediction_label'}, axis=1, inplace=True)
    return predictions['prediction_label']


if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    #df = load_data('churn_data_raw.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions.head())
