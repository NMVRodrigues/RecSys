import pandas as pd


def damped_mean(dataframe: pd.DataFrame,
                column_name: str,
                id_column: str,
                damping_factor: int=5,
                ):
    # Check if the column exists in the DataFrame
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Extract the specified column
    column = dataframe[column_name]

    # Compute the damped mean
    damped_mean_value = (dataframe.groupby(id_column)[column_name].sum() + damping_factor * dataframe[column_name].mean()) / (dataframe.groupby(id_column)[column_name].count() + damping_factor)

    dataframe[f'{column_name}_damped_mean'] = damped_mean_value

