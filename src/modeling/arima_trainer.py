import pandas as pd
from typing import Optional, Tuple

def select_arima_order(
    series: pd.Series,
    p_values: range = range(3),
    d_values: range = range(1, 2),
    q_values: range = range(3),
    seasonal_order: Tuple[int, int, int, int] = (0, 1, 0, 12),
) -> Tuple[Tuple[int, int, int], float]:
    pass