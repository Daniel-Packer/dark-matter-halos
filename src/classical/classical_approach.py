import pickle
from pathlib import Path

root_path = Path().resolve().parents[1]
data_path = root_path / 'data'
halos_path = data_path / 'halos.pkl'