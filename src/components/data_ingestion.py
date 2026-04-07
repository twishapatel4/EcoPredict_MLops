import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # The official OWID Carbon/Energy dataset (most stable source in 2026)
        self.data_url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

    def initiate_data_ingestion(self):
        logging.info("Initiating Data Ingestion using Our World in Data (OWID) Source")
        try:
            # 1. Read the remote CSV directly into Pandas
            logging.info(f"Downloading dataset from: {self.data_url}")
            df = pd.read_csv(self.data_url)
            
            logging.info("Dataset downloaded. Starting feature selection...")

            # 2. Select our "EcoPredict" Pillars
            # OWID columns: 
            # - gdp (total)
            # - population (total)
            # - energy_per_capita (kWh)
            # - co2_per_capita (Metric tons) <- OUR TARGET
            
            selected_columns = [
                'country', 'year', 'iso_code', 
                'gdp', 'population', 
                'energy_per_capita', 
                'co2_per_capita'
            ]
            
            df = df[selected_columns]

            # 3. Data Engineering: Final Clean
            # Remove rows where iso_code is null (removes continents/regions like 'World')
            df = df.dropna(subset=['iso_code'])
            
            # Remove rows where our target is missing
            df = df.dropna(subset=['co2_per_capita'])
            
            # Feature Construction: We need GDP per capita (OWID gives total GDP)
            df['gdp_per_capita'] = df['gdp'] / df['population']
            
            # Drop the raw columns we don't need for the final model
            df = df.drop(columns=['gdp', 'population'])

            # 4. Filter for our date range (2000 - 2022)
            df = df[(df['year'] >= 2000) & (df['year'] <= 2022)]

            # 5. Save the Cleaned Raw Data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            
            logging.info(f"Ingestion complete! Final dataset shape: {df.shape}")
            logging.info(f"Targets found: {df['co2_per_capita'].count()} rows")
            
            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()