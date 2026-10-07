import pandas as pd

def preprocess_ts(data_path: str="../../src/data/stores_sales_forecasting.csv"):
    dataset = pd.read_csv(
        data_path,
        encoding="latin-1",
        encoding_errors="replace")
    dataset['Order Date'] = pd.to_datetime(dataset['Order Date'])
    agg_sales = dataset.groupby(pd.Grouper(key='Order Date', freq='W'))['Sales'].sum()
    result_path = "data/aggregated_sales.csv"
    agg_sales.to_csv(result_path, index=True)
    return result_path