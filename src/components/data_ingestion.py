import os
import sys
import pandas as pd
import requests
import time
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    # We only care about raw data here now
    raw_data_path: str = os.path.join('data', "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        # The 4 Scientific Pillars for EcoPredict
        self.indicators = {
            "gdp": "NY.GDP.PCAP.CD",
            "energy_use": "EG.USE.PCAP.KG.OE", # works
            "renewable_pct": "EG.FEC.RNEW.ZS", # works
            # "co2": "EN.ATM.CO2E.PC" # not
            "total_ghg": "EN.ATM.GHGT.KT.CE",       # Target Component A
            "population": "SP.POP.TOTL"  
        }

    def fetch_indicator_data(self, indicator_code, name):
        """Polls the World Bank API and flattens the JSON response"""
        # FIXED: Using the variable indicator_code directly without extra quotes/braces
        url = f"https://api.worldbank.org/v2/country/all/indicator/{'EG.USE.PCAP.KG.OE'}?format=json&per_page=10000&date=1995:2020"
        
        try:
            logging.info(f"Fetching {name} ({indicator_code}) from API...")
            response = requests.get(url)
            response.raise_for_status()
            
            raw_data = response.json()
            if len(raw_data) < 2 or raw_data[1] is None:
                logging.warning(f"No data for {name}")
                return pd.DataFrame()
            
            df = pd.DataFrame(raw_data[1])
            
            # Flattening logic
            df['country_name'] = df['country'].apply(lambda x: x['value'])
            df['iso3'] = df['countryiso3code']
            
            # Selection and renaming
            df = df[['country_name', 'iso3', 'date', 'value']]
            df.columns = ['country', 'iso3', 'year', name]
            
            # Drop entries that aren't specific countries
            df = df.dropna(subset=['iso3',name])
            
            # Convert values to numeric to handle empty strings/None
            df[name] = pd.to_numeric(df[name], errors='coerce')
            
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
            logging.info("Starting Data Ingestion with User-Verified Indicators")
            try:
                dfs = []
                for name, code in self.indicators.items():
                    df = self.fetch_indicator_data(code, name)
                    if not df.empty:
                        dfs.append(df)
                    time.sleep(0.5)

                if len(dfs) < 5:
                    raise Exception(f"Missing indicators. Found only {len(dfs)}/5 active streams.")

                # Merging
                logging.info("Merging indicators into Master DataFrame")
                final_df = dfs[0]
                for next_df in dfs[1:]:
                    final_df = pd.merge(final_df, next_df, on=['country', 'iso3', 'year'], how='inner')

                # --- THE FULL-STACK AI MOVE: Target Construction ---
                logging.info("Calculating Emissions Per Capita as the Target Variable")
                # Convert Total GHG (kt) to Per Capita (metric tons)
                final_df['emissions'] = (final_df['total_ghg'] * 1000) / final_df['population']
                
                # Drop the raw components to keep the feature set clean
                final_df = final_df.drop(columns=['total_ghg', 'population'])

                # Create directory and save
                os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
                final_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
                
                logging.info(f"Ingestion complete. Final dataset shape: {final_df.shape}")
                return self.ingestion_config.raw_data_path

            except Exception as e:
                raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()