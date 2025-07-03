import pandas as pd

# === Time Key Creation Function ===
def create_time_keys(df):
    df = df.copy()
    df['Package Start Date'] = pd.to_datetime(df['Package Start Date'])

    days_since_sunday = (df['Package Start Date'].dt.dayofweek + 1) % 7
    week_start = df['Package Start Date'] - pd.to_timedelta(days_since_sunday, unit='D')
    df['Weeks_Key'] = week_start.dt.strftime('%Y-%m-%d')

    df['Months_Key'] = df['Package Start Date'].dt.to_period('M').astype(str)
    df['Quarters_Key'] = df['Package Start Date'].dt.to_period('Q').astype(str)
    df['Years_Key'] = df['Package Start Date'].dt.year.astype(str)

    return df