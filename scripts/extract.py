import requests  
import json     
from datetime import datetime, timezone
from pathlib import Path       

OUT_DIR = Path("./data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)  

def fetch_coingecko_top(n=100, vs_currency="usd"):
    """
    Llama a la API de CoinGecko y devuelve los n principales criptos en la moneda vs_currency.
    """
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": n,
        "page": 1,
        "sparkline": "false"
    }
    
    r = requests.get(url, params=params, timeout=30)  
    r.raise_for_status()  
    return r.json() 

def main():
    data = fetch_coingecko_top(n=250)

    # Use timezone-aware UTC datetime
    now_utc = datetime.now(timezone.utc)
    fname = OUT_DIR / f"coins_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("Saved:", fname)

if __name__ == "__main__":
    main()
