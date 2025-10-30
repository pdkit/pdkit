import numpy as np
from sklearn.neighbors import radius_neighbors_graph

def rpde(data, d=4, tau=50, eta=0.2, Tmax=1000):
    data = np.asarray(data)
    
    N = len(data)
    embd_length = N - (d-1)*tau
    if embd_length <= 0:
        return np.nan

    X = np.zeros((embd_length, d))
    for i in range(d):
        X[:, i] = data[i*tau : i*tau + embd_length]

    X = X - np.mean(X, axis=0)
    X = X / (np.std(X, axis=0) + 1e-6)

    close_return_times = []
    for i in range(embd_length-1):
        dists = np.linalg.norm(X[i+1:] - X[i], axis=1)
        close_idx = np.where(dists < eta)[0]

        if len(close_idx) > 0:
            close_return_times.append(close_idx[0]+1)

    if len(close_return_times) == 0:
        return np.nan

    close_return_times = np.array(close_return_times)
    close_return_times = close_return_times[close_return_times <= Tmax]

    if len(close_return_times) == 0:
        return np.nan

    if Tmax > 0:
        close_return_times = close_return_times[close_return_times <= Tmax]
        if len(close_return_times) == 0:
            return np.nan
    
    bins = np.arange(1, int(np.max(close_return_times)) + 2)
    hist, _ = np.histogram(close_return_times, bins=bins)
    
    hist = hist / np.sum(hist)
    
    hist = hist[hist > 0]
    
    rpde_val = -np.sum(hist * np.log(hist))
    
    rpde_val /= np.log(len(hist))

    return rpde_val

def rpde_fast(data, d=4, tau=50, eta=0.2, tmax=1000):
    data = np.asarray(data)
    data_length = len(data)
    embd_length = data_length - (d - 1) * tau

    if embd_length <= 0:
        return np.nan

    X = np.zeros((embd_length, d))
    for i in range(d):
        X[:, i] = data[i * tau : i * tau + embd_length]

    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0) + 1e-6

    graph = radius_neighbors_graph(X, radius=eta, mode='connectivity', include_self=False, n_jobs=1).tocsr()
    indptr = graph.indptr
    indices = graph.indices

    close_returns = []
    for i in range(embd_length - 1):
        start, end = indptr[i], indptr[i+1]
        if start == end:
            continue
        neighbors = indices[start:end]
        forward_neighbors = neighbors[neighbors > i]
        if forward_neighbors.size:
            dt = int(forward_neighbors.min() - i)
            if dt <= tmax:
                close_returns.append(dt)

    if not close_returns:
        return np.nan

    crt = np.array(close_returns, dtype=np.int32)
    
    bins = np.arange(1, tmax + 2)
    hist, _ = np.histogram(crt, bins=bins)
    hist = hist / np.sum(hist)
    
    N = len(hist)
    hist = hist[hist > 0]

    rpde_val = -np.sum(hist * np.log(hist))
    
    rpde_val /= np.log(N)
    
    return rpde_val



def rpde_forward_window(x, d=4, tau=50, eps=0.2, tmax=1000, metric="euclidean", standardize=True, step=1):
    """
    RPDE via direct forward search in a limited window of size tmax.
    Memory ~ O(d) + O(tmax). No neighbor lists/graphs.

    x : 1D signal
    d : embedding dimension
    tau : delay (in samples)
    eps : radius for 'close return'
    tmax : max forward time to consider (bins = 1..tmax)
    metric : 'euclidean' or 'chebyshev'
    standardize : z-score each embedding dimension
    step : stride to thin the embedded trajectory (e.g., 2 or 4 for long signals)
    """
    x = np.asarray(x, dtype=np.float32)
    N = len(x)
    embN_full = N - (d - 1) * tau
    if embN_full <= 1:
        return np.nan

    idx = np.arange(embN_full)[::max(1, int(step))]
    embN = len(idx)
    if embN <= 1:
        return np.nan

    X = np.empty((embN, d), dtype=np.float32)
    for k in range(d):
        Xi = x[k * tau : k * tau + embN_full]
        X[:, k] = Xi[idx]

    if standardize:
        mu = X.mean(axis=0, dtype=np.float64)
        sd = X.std(axis=0, dtype=np.float64) + 1e-6
        X = (X - mu) / sd

    hist = np.zeros(tmax, dtype=np.int64)
    for i in range(embN - 1):
        x_i = X[i]

        W = min(tmax, embN - 1 - i)
        if W <= 0:
            break
        block = X[i+1 : i+1+W]
        if metric == "chebyshev":
            dists = np.max(np.abs(block - x_i), axis=1)
        else: 
            diff = block - x_i
            dists = np.sqrt(np.sum(diff * diff, axis=1))
        hits = np.flatnonzero(dists <= eps)
        if hits.size:
            dt = int(hits[0] + 1)
            hist[dt-1] += 1

    if hist.sum() == 0:
        return np.nan

    p = hist.astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p)) 
    H_norm = H / np.log(len(hist))
    return float(H_norm)