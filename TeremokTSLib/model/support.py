"""
Support
---------
Support functions for ensemble model class.
"""

import pandas as pd
import numpy as np


# ----------------------------------------------------
# support methods


def check_spaces(dataframe):
    unique_dates = dataframe['date'].dt.date.unique()
    prev_date = unique_dates[0]
    for curr_date in unique_dates[1:]:
        expected_date = prev_date + pd.Timedelta(days=1)
        if curr_date != expected_date:
            missing_date = expected_date
            return missing_date
        prev_date = curr_date
    
def interpolate_spaces(df):
    while check_spaces(df) != None:
        missing_date = check_spaces(df)
        addon = df.head(1)
        addon['date'] = pd.to_datetime(missing_date) 
        print(f'found missing date: {missing_date}')
        for col in addon.columns:
            if col != 'date':
                if str(addon[col].values[0]).isdigit():
                    addon[col] = df[col].median()
                else:
                    addon[col] = max(set(df[col].tolist()), key=df[col].tolist().count)

        df = pd.concat([df, addon], ignore_index=True)
        df = df.sort_values(by='date').reset_index(drop=True)  