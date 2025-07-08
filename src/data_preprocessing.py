import pandas as pd
import numpy as np
from faker import Faker

def parse_ndc_data(package_file, product_file):
    """Parse NDC package.txt and product.txt files."""
    try:
        # Read tab-separated files
        package_df = pd.read_csv(package_file, sep='\t', usecols=['PRODUCTNDC', 'NDCPACKAGECODE'], nrows=1000)
        product_df = pd.read_csv(product_file, sep='\t', usecols=['PRODUCTNDC', 'LABELERNAME', 'PROPRIETARYNAME'], nrows=1000)
        
        # Rename columns for consistency
        package_df = package_df.rename(columns={'PRODUCTNDC': 'product_ndc', 'NDCPACKAGECODE': 'batch_id'})
        product_df = product_df.rename(columns={'PRODUCTNDC': 'product_ndc', 'LABELERNAME': 'manufacturer'})
        
        # Merge on product_ndc
        merged_df = pd.merge(package_df, product_df, on='product_ndc', how='inner')
        merged_df = merged_df[['batch_id', 'manufacturer', 'PROPRIETARYNAME']].dropna()
        
        return merged_df
    except Exception as e:
        print(f"Error parsing NDC data: {e}")
        return pd.DataFrame()

def generate_synthetic_features(data):
    """Add synthetic features for pricing, regions, and complaints."""
    fake = Faker()
    Faker.seed(42)  # Reproducible results
    n_samples = len(data)
    
    data['price'] = [round(fake.random.uniform(5.0, 50.0), 2) for _ in range(n_samples)]
    data['region'] = [fake.random_element(['Rural1', 'Rural2', 'Urban1', 'Urban2']) for _ in range(n_samples)]
    data['complaints'] = [fake.sentence(nb_words=10) for _ in range(n_samples)]
    
    # Feature engineering
    data['price_deviation'] = (data['price'] - data['price'].mean()) / data['price'].std()
    data['manufacturer_reliability'] = np.random.uniform(0.5, 1.0, n_samples)  # Mock reliability
    data['is_rural'] = data['region'].str.contains('Rural', case=False).astype(int)
    
    return data

def preprocess_data(package_file, product_file):
    """Combine NDC parsing and synthetic feature generation."""
    ndc_data = parse_ndc_data(package_file, product_file)
    if ndc_data.empty:
        print("No NDC data available, generating synthetic data")
        fake = Faker()
        Faker.seed(42)
        ndc_data = pd.DataFrame({
            'batch_id': [f'B{fake.unique.random_int(min=1000, max=9999)}' for _ in range(100)],
            'manufacturer': [fake.company() for _ in range(100)],
            'PROPRIETARYNAME': [fake.word().capitalize() for _ in range(100)]
        })
    processed_data = generate_synthetic_features(ndc_data)
    return processed_data