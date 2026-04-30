import pandas as pd
from sqlalchemy import create_engine, text
import urllib.parse
import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Database connection credentials
DB_USER = "root"
DB_PASS = "" # Password
DB_HOST = "127.0.0.1"
DB_NAME = "datathon_round1"

# Create Database engine
encoded_pass = urllib.parse.quote_plus(DB_PASS)
engine = create_engine(f"mysql+pymysql://{DB_USER}:{encoded_pass}@{DB_HOST}/{DB_NAME}")

CSV_DIR = os.path.join(os.path.dirname(__file__), "csv")

# IMPORT ORDER MATTERS due to Foreign Key constraints!
IMPORT_ORDER = [
    "geography",
    "customers",
    "products",
    "promotions",
    "orders",
    "order_items",
    "payments",
    "shipments",
    "returns",
    "reviews",
    "inventory",
    "sales",
    "sample_submission",
    "web_traffic"
]

def import_csvs():
    logging.info("Starting CSV data import to MySQL...")
    
    # Pre-check CSV directory
    if not os.path.exists(CSV_DIR):
        logging.error(f"Directory not found: {CSV_DIR}")
        return

    # To avoid FK checks causing issues during intermediate bulk inserts
    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        
        for table in IMPORT_ORDER:
            csv_path = os.path.join(CSV_DIR, f"{table}.csv")
            if not os.path.exists(csv_path):
                # Check for alternate naming (like sales_train.csv -> sales.csv)
                if table == "sales":
                    csv_path = os.path.join(CSV_DIR, "sales_train.csv")
                elif table == "sample_submission":
                    csv_path = os.path.join(CSV_DIR, "sales_test.csv")
                
                if not os.path.exists(csv_path):
                    logging.warning(f"File not found for table {table}, skipping...")
                    continue
            
            logging.info(f"Loading {os.path.basename(csv_path)} into table `{table}`...")
            
            try:
                # Read CSV
                df = pd.read_csv(csv_path, low_memory=False)
                
                # Convert dates to valid format. Check known date columns
                for col in df.columns:
                    if "date" in col.lower():
                        try:
                            df[col] = pd.to_datetime(df[col], dayfirst=False).dt.date
                        except Exception:
                            pass
                
                # Write to database (append mode so we don't recreate tables without constraints)
                df.to_sql(name=table, con=conn, if_exists="append", index=False, chunksize=10000, method="multi")
                logging.info(f"  -> Successfully imported {len(df)} rows to `{table}`")
                
            except Exception as e:
                logging.error(f"  -> Failed to import {table}: {e}")

        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
    
    logging.info("Import process complete!")

if __name__ == "__main__":
    import_csvs()
