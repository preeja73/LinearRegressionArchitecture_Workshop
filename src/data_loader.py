import pandas as pd
import requests
import sqlite3

def load_csv(path):
    """Load data from a CSV file."""
    return pd.read_csv(path)

def load_api(url):
    """Load data from an API endpoint."""
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["result"], columns=["name"])

def load_db(db_path, table_name):
    """Load data from a SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df