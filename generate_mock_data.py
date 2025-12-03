"""
Mock Data Generator for TrendSpotter
Generates realistic AdTech CSV data with intentional anomalies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_ad_traffic_data():
    """
    Generate mock AdTech data with realistic patterns and anomalies
    """
    np.random.seed(42)
    
    # Generate 30 days of data
    start_date = datetime(2024, 11, 1)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Campaign IDs
    campaigns = ['CAMP_001', 'CAMP_002', 'CAMP_003', 'CAMP_004']
    
    # Locations
    locations = ['New York', 'Los Angeles', 'Miami', 'Chicago', 'Seattle']
    
    data = []
    
    for date in dates:
        for campaign in campaigns:
            for location in locations:
                # Base metrics with realistic patterns
                base_impressions = np.random.randint(50000, 150000)
                base_clicks = int(base_impressions * np.random.uniform(0.02, 0.05))
                base_spend = np.random.uniform(500, 2000)
                
                # Introduce anomalies
                # Anomaly 1: Massive drop in Miami on Nov 10-12 (hurricane impact)
                if location == 'Miami' and 10 <= date.day <= 12:
                    base_impressions = int(base_impressions * 0.1)
                    base_clicks = int(base_clicks * 0.05)
                    base_spend = base_spend * 0.2
                
                # Anomaly 2: Unusual spike in Seattle on Nov 20 (Black Friday prep)
                if location == 'Seattle' and date.day == 20:
                    base_impressions = int(base_impressions * 3.5)
                    base_clicks = int(base_clicks * 4.2)
                    base_spend = base_spend * 3.8
                
                # Anomaly 3: Campaign 003 has budget issues on Nov 15
                if campaign == 'CAMP_003' and date.day == 15:
                    base_spend = base_spend * 5.5
                    base_clicks = int(base_clicks * 0.3)
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'campaign_id': campaign,
                    'clicks': max(1, base_clicks),
                    'impressions': max(100, base_impressions),
                    'spend': round(base_spend, 2),
                    'location': location
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('ad_traffic_data.csv', index=False)
    print(f"Successfully generated ad_traffic_data.csv with {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nIntroduced anomalies:")
    print("1. Miami: Significant traffic drop (Nov 10-12)")
    print("2. Seattle: Unusual spike (Nov 20)")
    print("3. CAMP_003: Budget overspend (Nov 15)")

if __name__ == "__main__":
    generate_ad_traffic_data()

