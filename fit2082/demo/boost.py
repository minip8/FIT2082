import numpy as np
from numba import njit, prange

# from numba.typed import List
# from numba.types import f4, i4

# ==============================================================================
# == Boosting Class ============================================================
# ==============================================================================

class NewHashBoost:

    # num_classes : num classes
    # num_pairs_per_hash : num bits per hash; uses pairs of points to determine split points/partitions
    # lr : boosting learning rate
    # max_num_hashes : maximum number of total hashes in model
    def __init__(self, num_classes, num_pairs_per_hash = 8, lr = 0.1, max_num_hashes = 100):

        self.num_classes = np.int32(num_classes)
        self.num_pairs_per_hash = np.int32(num_pairs_per_hash)

        self.lr = np.float32(lr)

        # these "buffers" hold the numerator and denominator used to compute the final output of each of our boosted models
        # by maintaining these separately we can always compute the exact output
        # more stable

        self.buffer_residuals = np.zeros((max_num_hashes, 2 ** num_pairs_per_hash, num_classes), dtype = np.float32)
        self.buffer_hessian = np.zeros((max_num_hashes, 2 ** num_pairs_per_hash, num_classes), dtype = np.float32)

        self.logits0 = np.zeros((max_num_hashes, 2 ** num_pairs_per_hash, num_classes), dtype = np.float32) # logits, but derived from numerator and denominator buffers only
        
        self.feature_indices = np.zeros((max_num_hashes, num_pairs_per_hash), dtype = np.int32)
        self.midpoints = np.zeros((max_num_hashes, num_pairs_per_hash), dtype = np.float32)

        self.num_rounds = 0

    # fit a (mini)batch of data, i.e., a new round of boosting
    # X : input features, size [n, p] np.float32
    # Y : class labels, size [n] np.int32
    def fit_batch(self, X, Y):

        _fit_batch0(
            X = X,
            Y = Y,
            buffer_residuals = self.buffer_residuals,
            buffer_hessian = self.buffer_hessian,
            feature_indices = self.feature_indices,
            midpoints = self.midpoints,
            num_classes = self.num_classes,
            num_rounds = self.num_rounds,
            lr = self.lr,
            num_pairs_per_hash = self.num_pairs_per_hash,
            logits0 = self.logits0,
        )

        self.num_rounds += 1

    # raw logits (output of final boosting round)
    def predict(self, X):

        return \
        _predict_multi0(
            X = X,
            feature_indices = self.feature_indices,
            midpoints = self.midpoints,
            num_rounds = self.num_rounds,
            num_classes = self.num_classes,
            logits0 = self.logits0,
        )

    # raw logits (output of all boosting rounds individually)
    def predict_all(self, X):

        return \
        _predict_multi_all0(
            X = X,
            feature_indices = self.feature_indices,
            midpoints = self.midpoints,
            num_rounds = self.num_rounds,
            num_classes = self.num_classes,
            logits0 = self.logits0,
        )

    # softmax-ed probabilities
    def predict_proba(self, X):

        return _softmax0(self.predict(X))

# ==============================================================================
# == Supporting Functions ======================================================
# ==============================================================================

# -- one hot encoding ----------------------------------------------------------

@njit("i4[:,:](i4[:],i4)", fastmath = True, cache = True)
def _one_hot0(Y, num_classes):

    num_examples = Y.shape[0]

    Z = np.zeros((num_examples, num_classes), dtype = np.int32)

    for i in range(num_examples):
        Z[i, Y[i]] = 1

    return Z

# -- cross entropy -------------------------------------------------------------

@njit("f4[:](f4[:,:],i4[:])", fastmath = True, cache = True)
def _cross_entropy(Z, Y):

    n, _p = Z.shape

    ce = np.zeros(n, dtype = np.float32)

    for i in range(n):

        y = Y[i]
        ce[i] = -np.log(Z[i, y])

    return ce

# -- softmax -------------------------------------------------------------------

@njit("f4[:,:](f4[:,:])", fastmath = True, cache = True)
def _softmax0(X):

    num_examples, k = X.shape

    Z = np.empty_like(X, dtype = np.float32)

    for i in range(num_examples):

        max_value = X[i, 0]
        for c in range(1, k):
            max_value = max(max_value, X[i, c])
        
        sum_exp = np.float32(0)
        for c in range(k):
            Z[i, c] = np.exp(X[i, c] - max_value)
            sum_exp += Z[i, c]
        
        for c in range(k):
            Z[i, c] /= sum_exp

    return Z

# -- convert base 2 -> base 10 -------------------------------------------------

@njit("i4[:](b1[:,:])", fastmath = True, cache = True)
def _base10_0(b2):

    n, p = b2.shape

    b10 = np.zeros(n, dtype = np.int32)

    for i in range(n):
        v = 0
        for j in range(p):
            if b2[i, j]:
                v += (1 << j)
        b10[i] = v

    return b10

# -- predict single hash -------------------------------------------------------

@njit("void(f4[:,:],i4[:],f4[:],f4[:,:],f4[:,:])", fastmath = True, cache = True)
def _predict_single0(X, feature_indices, midpoints, logits0, buffer):

    n = X.shape[0]
    p = feature_indices.shape[0]
    k = logits0.shape[-1]

    for i in range(n):
        ix = 0
        for j in range(p):
            if X[i, feature_indices[j]] <= midpoints[j]:
                ix += (1 << j)

        for c in range(k):
            buffer[i, c] += logits0[ix, c]

# -- predict multiple hashes ---------------------------------------------------

@njit("f4[:,:](f4[:,:],i4[:,:],f4[:,:],i4,i4,f4[:,:,:])", fastmath = True, cache = True, parallel = True)
def _predict_multi0(X, feature_indices, midpoints, num_rounds, num_classes, logits0):

    n = X.shape[0]

    P = np.zeros((n, num_classes), dtype = np.float32)

    for r in prange(num_rounds):  # ty: ignore[not-iterable]

        buffer = np.zeros((n, num_classes), dtype = np.float32)

        _predict_single0(
            X = X,
            feature_indices = feature_indices[r],
            midpoints = midpoints[r],
            logits0 = logits0[r],
            buffer = buffer,
        )

        P += buffer

    return P

# -- predict multiple hashes (keep all results separately) ---------------------

@njit("f4[:,:,:](f4[:,:],i4[:,:],f4[:,:],i4,i4,f4[:,:,:])", fastmath = True, cache = True, parallel = True)
def _predict_multi_all0(X, feature_indices, midpoints, num_rounds,  num_classes, logits0):

    n = X.shape[0]

    P = np.zeros((num_rounds + 1, X.shape[0], num_classes), dtype = np.float32)
    p0 = np.float32(np.log(1 / num_classes))

    for i in range(n):
        for j in range(num_classes):
            P[0, i, j] = p0

    buffer = np.zeros((num_rounds, n, num_classes), dtype = np.float32)

    for r in prange(num_rounds):  # ty: ignore[not-iterable]

        _predict_single0(
            X = X,
            feature_indices = feature_indices[r],
            midpoints = midpoints[r],
            logits0 = logits0[r],
            buffer = buffer[r],
        )

        P[r + 1, :] = buffer[r]

    return P

# -- update single hash --------------------------------------------------------

@njit("void(f4[:,:],f4[:,:],f4[:,:],i4[:],f4[:],i4,f4[:,:],f4[:,:],f4,f4[:,:],f4[:,:],f4[:,:],b1[:])", fastmath = True, cache = True)
def _update_single0(X, buffer_residuals, buffer_hessian, feature_indices, midpoints, num_classes, residuals, hessian, lr, logits0, RR, HH, visited):

    num_examples = X.shape[0]

    p = feature_indices.shape[0]

    s2 = np.empty((num_examples, p), dtype = np.bool_)
    for i in range(num_examples):
        for j in range(p):
            s2[i, j] = X[i, feature_indices[j]] <= midpoints[j]
    
    s10 = _base10_0(s2)

    RR.fill(np.float32(0.0))
    HH.fill(np.float32(0.0))
    visited.fill(False)

    for i in range(num_examples):

        s10_i = s10[i]

        for k in range(num_classes):

            RR[s10_i, k] += residuals[i, k]
            HH[s10_i, k] += hessian[i, k]

        visited[s10_i] = True

    for i in range(buffer_residuals.shape[0]):

        if visited[i]:

            for k in range(num_classes):

                # basically here we are maintaining the numerator and denominator separately
                # allows for computing the exact value every time
                # more stable

                buffer_residuals[i, k] -= RR[i, k]
                buffer_hessian[i, k] += HH[i, k]

                g = buffer_residuals[i, k]
                h = buffer_hessian[i, k]

                value = g / (h + np.float32(1e-6)) * lr
                logits0[i, k] = value

# -- predict all hashes --------------------------------------------------------

@njit("void(f4[:,:],f4[:,:,:],f4[:,:,:],i4[:,:],f4[:,:],i4,f4[:,:],f4[:,:],f4,i4,f4[:,:,:])", fastmath = True, cache = True, parallel = True)
def _update_multi0(X, buffer_residuals, buffer_hessian, feature_indices, midpoints, num_classes, residuals, hessian, lr, num_rounds, logits0):

    hash_size = buffer_residuals.shape[1]

    for r in prange(num_rounds):  # ty: ignore[not-iterable]

        RR0 = np.empty((hash_size, num_classes), dtype = np.float32)
        HH0 = np.empty((hash_size, num_classes), dtype = np.float32)
        visited0 = np.empty(hash_size, dtype = np.bool_)

        _update_single0(
            X = X,
            buffer_residuals = buffer_residuals[r],
            buffer_hessian = buffer_hessian[r],
            feature_indices = feature_indices[r],
            midpoints = midpoints[r],
            num_classes = num_classes,
            residuals = residuals,
            hessian = hessian,
            lr = lr,
            logits0 = logits0[r],
            RR = RR0,
            HH = HH0,
            visited = visited0,
        )

# ==============================================================================
# ==============================================================================
# ==============================================================================

# -- form pairs of points for establishing partitions --------------------------

@njit("i4[:,:](i4[:],i4[:],i4)", fastmath = True, cache = True)
def _pair(Y, order, num_pairs):

    num_examples = Y.shape[0]

    num_found = 0

    pairs = np.zeros((num_pairs, 2), dtype = np.int32)
    pairs_classes = np.zeros((num_pairs, 2), dtype = np.int32)
    status = np.zeros(num_pairs)

    pointer_data = 0
    while num_found < num_pairs:

        new = order[pointer_data]
        new_Y = Y[new]

        pointer_pair = num_found
        processed = False
        
        while not processed and pointer_pair < num_pairs:

            if status[pointer_pair] == 0:
                status[pointer_pair] = 1
                pairs_classes[pointer_pair, 0] = new_Y
                pairs[pointer_pair, 0] = new
                processed = True
            elif status[pointer_pair] == 1:
                if pairs_classes[pointer_pair, 0] != new_Y:
                    status[pointer_pair] = 2
                    pairs_classes[pointer_pair, 1] = new_Y
                    pairs[pointer_pair, 1] = new
                    num_found += 1
                    processed = True
            
            pointer_pair += 1

        pointer_data += 1
        if pointer_data >= num_examples:
            pointer_data = 0
    
    return pairs

# -- pairs -> partitions -------------------------------------------------------

@njit("Tuple((i4[:],f4[:]))(f4[:,:],i4[:,:])", fastmath = True, cache = True)
def _assign(X, pairs_indices):

    _num_examples, num_features = X.shape

    num_pairs = pairs_indices.shape[0]

    ix = np.random.randint(0, num_features, num_pairs).astype(np.int32)

    candidates = np.zeros((num_pairs, 2), dtype = np.float32)

    for i in range(num_pairs):

        ia = pairs_indices[i, 0]
        ib = pairs_indices[i, 1]

        Xa = X[ia, ix[i]]
        Xb = X[ib, ix[i]]

        candidates[i] = Xa, Xb

    midpoints = candidates.sum(-1) / 2

    return ix, midpoints

# ==============================================================================
# ==============================================================================
# ==============================================================================

# -- fit a new hash (single round of boosting) ---------------------------------

@njit("Tuple((i4[:],f4[:]))(f4[:,:],i4[:],f4[:,:],f4[:,:],i4,f4[:,:],f4[:,:],f4,i4,f4[:,:],f4[:,:])", fastmath = True, cache = True)
def _fit_new_hash0(X, Y, buffer_residuals, buffer_hessian, num_classes, residuals, hessian, lr, num_pairs_per_hash, logits0, pred_prob):

    num_examples = X.shape[0]

    ce = _cross_entropy(pred_prob, Y)
    order = np.argsort(ce)[::-1].astype(np.int32)

    feature_indices, midpoints = _assign(X, _pair(Y = Y, order = order, num_pairs = num_pairs_per_hash))

    s2 = X[..., feature_indices] <= midpoints
    s10 = _base10_0(s2)

    RR = np.zeros_like(buffer_residuals, dtype = np.float32)
    HH = np.zeros_like(buffer_hessian, dtype = np.float32)

    visited = np.zeros(buffer_residuals.shape[0], dtype = np.bool_)

    for i in range(num_examples):

        s10_i = s10[i]

        for k in range(num_classes):

            RR[s10_i, k] += residuals[i, k]
            HH[s10_i, k] += hessian[i, k]

        visited[s10_i] = True

    for i in range(buffer_residuals.shape[0]):

        if visited[i]:

            for k in range(num_classes):

                buffer_residuals[i, k] -= RR[i, k]
                buffer_hessian[i, k] += HH[i, k]

                g = buffer_residuals[i, k]
                h = buffer_hessian[i, k]

                value = g / (h + np.float32(1e-6)) * lr
                logits0[i, k] = value

    return feature_indices, midpoints

# -- fit new (mini)batch of data: (a) update existing; and (b) fit new ---------

@njit("void(f4[:,:],i4[:],f4[:,:,:],f4[:,:,:],i4[:,:],f4[:,:],i4,i4,f4,i4,f4[:,:,:])", fastmath = True, cache = True)
def _fit_batch0(X, Y, buffer_residuals, buffer_hessian, feature_indices, midpoints, num_classes, num_rounds, lr, num_pairs_per_hash, logits0):

    # get predictions for current (mini)batch

    log_odds = \
    _predict_multi0(
        X = X,
        feature_indices = feature_indices,
        midpoints = midpoints,
        num_rounds = num_rounds,
        num_classes = num_classes,
        logits0 = logits0,
    )

    Z = _one_hot0(Y, num_classes = num_classes)

    probabilities = _softmax0(log_odds)

    # get residuals

    residuals = (probabilities - Z).astype(np.float32)
    hessian = (probabilities * (1 - probabilities) + np.float32(1e-3)).astype(np.float32)

    # update existing models

    if num_rounds > 0:

        _update_multi0(
            X = X,
            buffer_residuals = buffer_residuals,
            buffer_hessian = buffer_hessian,
            feature_indices = feature_indices,
            midpoints = midpoints,
            num_classes = num_classes,
            residuals = residuals,
            hessian = hessian,
            lr = lr,
            num_rounds = num_rounds,
            logits0 = logits0,
        )

    # fit a new model to the end

    feature_indices[num_rounds], midpoints[num_rounds] = \
    _fit_new_hash0(
        X = X,
        Y = Y,
        buffer_residuals = buffer_residuals[num_rounds],
        buffer_hessian = buffer_hessian[num_rounds],
        num_classes = num_classes,
        residuals = residuals,
        hessian = hessian,
        lr = lr,
        num_pairs_per_hash = num_pairs_per_hash,
        logits0 = logits0[num_rounds],
        pred_prob = probabilities,
    )