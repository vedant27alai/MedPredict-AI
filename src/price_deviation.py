import pandas as pd
import os
import re

# Load official price list once at top
price_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data','processed', 'nppaipdms.csv'))

def clean_price_string(price_str):
    """ Extract float from ₹ 0.28(1 Tablet) """
    match = re.search(r'₹\s?([\d.]+)', price_str)
    if match:
        return float(match.group(1))
    return None

def load_price_reference():
    df = pd.read_csv(price_file_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'tablescraper-selected-row': 'medicine_name',
        'tablescraper-selected-row 2': 'dosage_form',
        'tablescraper-selected-row 5': 'official_price'
    })
    df['medicine_name'] = df['medicine_name'].str.strip().str.lower()
    df['official_price_clean'] = df['official_price'].apply(clean_price_string)
    df = df.dropna(subset=['official_price_clean'])
    return df[['medicine_name', 'dosage_form', 'official_price_clean']]

# Cache the data
official_price_df = load_price_reference()

def get_official_price(medicine_name: str) -> float:
    """
    Return the official price per unit if available.
    """
    med = medicine_name.strip().lower()
    match = official_price_df[official_price_df['medicine_name'] == med]
    if not match.empty:
        return match.iloc[0]['official_price_clean']
    return None

def calculate_price_deviation(user_price_per_unit: float, medicine_name: str) -> float:
    """
    Compare user price with official price and return deviation in percentage.
    """
    official = get_official_price(medicine_name)
    if official is None or official == 0:
        return None
    deviation = ((user_price_per_unit - official) / official) * 100
    return round(deviation, 2)
