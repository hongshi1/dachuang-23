import numpy as np
from sklearn.metrics import pairwise_distances


def SMOTEND(data_set, ide_ratio=1, k=5):
    np.random.seed(1)

    n = data_set.shape[0]
    m = np.sum(data_set[:, -1] != 0)

    if m >= (n - m):
        return np.array([])

    mino_sam = data_set[data_set[:, -1] != 0, :]
    mino_sam_x = mino_sam[:, :-1]

    g_num = int((n - m) * ide_ratio - m)
    if g_num < m:
        index_ori = 1
        m = int(g_num)
    else:
        index_ori = int(g_num / m)

    D = pairwise_distances(mino_sam_x)
    np.fill_diagonal(D, 0)
    idx = np.argsort(D, axis=1)[:, :k]

    syn_samples = np.zeros((m * index_ori, data_set.shape[1]))
    count = 0

    for i in range(m):
        index = index_ori

        while index:
            if k <= idx.shape[1]:
                nn = np.random.permutation(idx[i, :])[:1]
            else:
                temp0 = idx.shape[1]
                temp = np.random.permutation(temp0)[:1]
                nn = idx[i, temp]

            x_nn = mino_sam_x[nn, :]
            x_i = mino_sam_x[i, :]
            x_syn = x_i + np.random.rand() * (x_nn - x_i)

            d1 = np.linalg.norm(x_syn - x_i)
            d2 = np.linalg.norm(x_syn - x_nn)
            y_syn = (d2 * mino_sam[i, -1] + d1 * mino_sam[nn, -1]) / (d1 + d2)

            syn_samples[count, :] = np.hstack([x_syn.flatten(), y_syn])
            count += 1
            index -= 1

    syn_samples = syn_samples[~np.isnan(syn_samples[:, -1]), :]
    return syn_samples
