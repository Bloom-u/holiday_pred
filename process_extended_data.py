"""
Script to load and process the extended dataset with holiday features
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.insert(0, './src')

from holiday_feature import HolidayFeatureEngine

def process_extended_data():
    """Process the extended dataset with holiday features"""
    print("Loading extended dataset...")
    df = pd.read_csv('./data/all_data_2012_2024.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['GCRQ'].min()} to {df['GCRQ'].max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Rename columns to match the expected format
    df = df.rename(columns={
        'GCRQ': '日期',
        'KC_DL': '客车当量',
        'HC_DL': '货车当量',
        'KC_CS': '客车车速',
        'HC_CS': '货车车速',
        'JDC_DL': '机动车当量',
        'JDC_CS': '机动车车速',
        'XZQHMC': '行政区划名称'
    })
    
    # Convert date column
    df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d')
    
    # Select relevant columns (keeping only traffic volume columns)
    df_selected = df[['日期', '客车当量', '货车当量', '机动车当量']].copy()
    
    # Remove any rows with missing values in the target column
    df_selected = df_selected.dropna(subset=['机动车当量'])
    
    print(f"Selected dataset shape: {df_selected.shape}")
    print(f"Date range after cleaning: {df_selected['日期'].min()} to {df_selected['日期'].max()}")
    
    # Generate holiday features
    print("\nGenerating holiday features...")
    engine = HolidayFeatureEngine()
    df_with_features = engine.create_features(df_selected, date_column='日期')
    
    # Save the processed data
    output_path = './data/extended_df_with_features.pkl'
    df_with_features.to_pickle(output_path)
    print(f"\nProcessed data saved to: {output_path}")
    
    print(f"\nFinal dataset shape: {df_with_features.shape}")
    print(f"Date range: {df_with_features['日期'].min()} to {df_with_features['日期'].max()}")
    
    # Print some statistics
    print(f"\nTraffic volume statistics:")
    print(f"Mean: {df_with_features['机动车当量'].mean():.2f}")
    print(f"Std: {df_with_features['机动车当量'].std():.2f}")
    print(f"Min: {df_with_features['机动车当量'].min():.2f}")
    print(f"Max: {df_with_features['机动车当量'].max():.2f}")
    
    # Count holiday types
    print(f"\nHoliday type distribution:")
    holiday_counts = df_with_features['holiday_type'].value_counts().sort_index()
    for holiday_type, count in holiday_counts.items():
        type_names = {
            0: 'Workday',
            1: 'Weekend', 
            2: "New Year's Day",
            3: 'Spring Festival',
            4: 'Tomb-sweeping Day',
            5: 'Labour Day',
            6: 'Dragon Boat Festival',
            7: 'Mid-autumn Festival',
            8: 'National Day',
            9: 'Adjusted Workday'
        }
        name = type_names.get(holiday_type, f'Other({holiday_type})')
        print(f"  {name} (type {holiday_type}): {count} days")
    
    return df_with_features

if __name__ == '__main__':
    df_extended = process_extended_data()