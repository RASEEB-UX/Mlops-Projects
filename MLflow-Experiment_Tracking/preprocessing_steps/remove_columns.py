import pandas as pd


def del_columns(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    existing_columns = [col for col in df.columns if col in col_list]
    return df.drop(columns=existing_columns)