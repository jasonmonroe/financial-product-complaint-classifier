# src/preprocessing.py

import pandas as pd

import config
import seeder


def load_data(seed_data: bool = False) -> pd.DataFrame:
    # Load the data from the csv file.
    df = pd.read_csv(csv_file_path + csv_file_name)

    if seed_data:
        seeded_data = seeder.run()
        df = merge_data(df, seeded_data)

    return df

def merge_data(df: pd.DataFrame, seeded_data: list) -> pd.DataFrame:
    # Merge the seeded data with the original data.
    df_seeded = pd.DataFrame(seeded_data, columns=df.columns)
    df_merged = pd.concat([df, df_seeded], ignore_index=True)

    return df_merged