import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from ml.data_loader import load_crypto_data
from ml.feature_engineer import create_target

def visualize_price_history(crypto_id='bitcoin', days=7):
    """
    Visualiza el historial de precios y targets.
    """
    df = load_crypto_data(crypto_id=crypto_id, days=days)
    df_with_target = create_target(df)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(df_with_target['last_updated'], df_with_target['current_price'], marker='o')
    plt.title(f'{crypto_id.upper()} - Historial de Precios')
    plt.ylabel('Precio (USD)')
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 1, 2)
    colors = ['red' if x == 0 else 'green' for x in df_with_target['target']]
    plt.scatter(df_with_target['last_updated'], df_with_target['target'], c=colors, s=100)
    plt.title('Target: 0=BAJA, 1=SUBE')
    plt.ylabel('Dirección')
    plt.yticks([0, 1], ['BAJA', 'SUBE'])
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    viz_dir = Path('visualizations')
    viz_dir.mkdir(exist_ok=True)
    filepath = viz_dir / f'{crypto_id}_history.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Visualización guardada en: {filepath}")
    plt.close()

if __name__ == '__main__':
    print("Generando visualización de precios...")
    visualize_price_history()
    print("Completo!")

