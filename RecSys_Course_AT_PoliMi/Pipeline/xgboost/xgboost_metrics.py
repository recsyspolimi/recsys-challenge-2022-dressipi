from typing import Tuple

import numpy as np
import xgboost as xgb


# TODO custom metric for xgboost (see documentation)


def mean_reciprocal_rank(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    predt = (np.asarray(r).nonzero()[0] for r in predt)
    return 'MRR', float(np.mean([1. / (r[0] + 1) if r.size else 0. for r in predt]))
