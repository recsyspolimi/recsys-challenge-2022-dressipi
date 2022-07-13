import numpy as np

def set_zeros_row_col(dataMat, rows, cols):
    for row in rows:
        row_start = dataMat.indptr[row]
        row_end = dataMat.indptr[row+1]
        mask_data = np.isin(dataMat.indices[row_start:row_end], cols)
        dataMat.data[row_start:row_end][mask_data] = 0
    dataMat.eliminate_zeros()