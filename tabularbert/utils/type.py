import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler

# Type
ArrayLike = np.ndarray | pd.DataFrame
Scaler = QuantileTransformer | StandardScaler | MinMaxScaler