import pandas as pd
from pathlib import Path
import json
from datetime import datetime

RAW_DIR = Path("./data/raw")
PROC_DIR = Path("./data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def load_latest_raw():
    """
    Carga el último archivo JSON de la carpeta raw.
    """
    files = sorted(RAW_DIR.glob("coins_*.json")) 
    if not files:
        raise FileNotFoundError("No raw files found in data/raw")
    return json.loads(files[-1].read_text(encoding="utf-8"))  

def transform(raw):
    """
    Limpieza y transformación del JSON a DataFrame listo para cargar.
    """
    df = pd.DataFrame(raw)
    
    df = df[['id','symbol','name','current_price','market_cap','total_volume','price_change_percentage_24h','last_updated']]

    df['last_updated'] = pd.to_datetime(df['last_updated'])

    df['date'] = df['last_updated'].dt.date

    df['market_cap_millions'] = df['market_cap'] / 1e6

    df = df.dropna(subset=['current_price'])
    
    return df

def main():
    raw = load_latest_raw()
    df = transform(raw)

    out_file = PROC_DIR / f"coins_processed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.parquet"
    df.to_parquet(out_file, index=False)
    
    print("Processed saved:", out_file)

if __name__ == "__main__":
    main()
