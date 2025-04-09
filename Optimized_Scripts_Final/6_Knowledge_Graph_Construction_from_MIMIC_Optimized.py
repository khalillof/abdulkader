import pandas as pd
import numpy as np
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start timing the script execution
start_time = time.time()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load a small sample of the dataset instead of the entire file
def load_data(filepath, sample_size=1000):
    if os.path.exists(filepath):
        return pd.read_csv(filepath, nrows=sample_size)  # Load only first 1000 rows for efficiency
    else:
        logging.warning(f"File {filepath} not found! Returning empty DataFrame.")
        return pd.DataFrame()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, GATConv
from torch_geometric.data import Data, HeteroData
from IPython.display import HTML, display
# End timing
end_time = time.time()
logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")
