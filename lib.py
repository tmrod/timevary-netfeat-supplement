import numpy as np
import scipy
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import gmean

def sbm_pp_parameters(k, p, q, sigma=0):
    """Returns the affinity matrix for a planted partition matrix. If a nonzero sigma is passed, the parameter matrix will be perturbed by Gaussian noise.

    Parameters
    ----------
    k : int
        Number of blocks
    p : float
        Intracluster edge probability
    q : float
        Intercluster edge probability
    sigma : float (optional)
        Stdev of perturbation

    Returns
    -------
    omega : ndarray of shape (k,k)
        Affinity matrix for an SBM
    """
    omega = (p-q)*np.eye(k) + q*np.ones([k, k])

    pert = sigma*np.random.randn(*omega.shape)
    pert = np.tril(pert) + np.tril(pert, -1).T
    omega += pert
    np.maximum(omega, 0, omega)

    return omega

def label_nodes(sizes):
    """Assign labels to nodes for each group

    Parameters
    ----------
    sizes : list of ints
        List of block sizes

    Returns
    -------
    list of ints
        List of node labels from `0` to `k-1`
    """
    labels = np.concatenate([tup[0] * np.ones(tup[1])
                             for tup
                             in enumerate(sizes)]).astype(int)
    return list(labels)

def partition_indicator(labels):
    """Partition indicator matrix

    Parameters
    ----------
    labels : list of ints
        List of node labels, from 0 to k-1

    Returns
    -------
    G : scipy csr matrix of shape (N,k)
        The partition indicator matrix
    """
    partition_vec = np.array(labels).astype(int)
    nr_nodes = partition_vec.size
    k = np.max(partition_vec) + 1

    G = scipy.sparse.coo_matrix((np.ones(nr_nodes), (np.arange(nr_nodes), partition_vec)),
                                shape=(nr_nodes, k)).tocsr()

    return G

def expected_sbm_matrix(labels, omega):
    """Returns the expected adjacency matrix of an SBM

    Parameters
    ----------
    labels : list of ints
        Node labels
    omega : ndarray of shape (k,k)
        Block affinity matrix

    Returns
    -------
    ndarray of shape (N,N)
        The expected model's adjacency matrix
    """
    G = partition_indicator(labels).toarray()
    return  G @ omega @ G.T

def bernoulli_adjacency_matrix(E):
    """Draws an adjacency matrix via a Bernoulli process on `E`

    Parameters
    ----------
    E : ndarray of shape (N,N)
        Edge probabilities: only lower triangle is used

    Returns
    -------
    A : ndarray of shape (N,N)
        An instance of the specified model's adjacency matrix
    """
    A = np.random.binomial(1, E)
    A = np.tril(A) + np.tril(A, -1).T
    return A

def laplacian(A):
    """Returns the graph Laplacian of an adjacency matrix

    Parameters
    ----------
    A : ndarray of shape (N,N)
        Adjacency matrix
    
    Returns
    -------
    L : ndarray of shape (N,N)
        Laplacian matrix
    """
    D = np.diag(np.sum(A, axis=1))
    return D - A
    
def filter_matrix(S, h):
    """Graph filter matrix from a shift operator and coefficients

    Parameters
    ----------
    S : ndarray of shape (N,N)
        Graph shift operator
    h : list of floats
        List of filter coefficients

    Returns
    -------
    H : ndarray of shape (N,N)
        The graph filter matrix
    """
    return sum([coeff*np.linalg.matrix_power(S, idx)
                for idx, coeff
                in enumerate(h)])

def empirical_covariance(system, excitation, m):
    """Applies a random diffusion process to an input signal

    Parameters
    ----------
    system : function of type () -> ndarray of shape (N,N)
        Function that generates random graph filters
    excitation : function of type () -> ndarray of shape(N,)
        Function that generates excitation signals
    m : int
        Number of samples to draw

    Returns
    -------
    C : ndarray of shape (N,N)
        Empirical covariance matrix as outlined in the paper
    """
    observations = [system() @ excitation() for _ in range(m)]
    return np.cov(np.array(observations).T)

def empirical_covariance_multiple(system, excitation, ms):
    """Applies a random diffusion process to an input signal, for multiple observation counts

    Parameters
    ----------
    system : function of type () -> ndarray of shape (N,N)
        Function that generates random graph filters
    excitation : function of type () -> ndarray of shape(N,)
        Function that generates excitation signals
    ms : list of ints
        Number of samples to draw

    Returns
    -------
    Cs : list of ndarray of shape (N,N)
        Empirical covariance matrix as outlined in the paper
    """
    observations = [system() @ excitation() for _ in range(max(ms))]

    Cs = [np.cov(np.array(observations[0:m]).T) for m in ms]
    return Cs

def order_selection(C, threshold):
    """Naive threshold model order estimator

    Parameters
    ----------
    C : ndarray of shape (N,N)
        Covariance matrix
    threshold : float

    Returns
    -------
    int
        Number of eigenvalues of `C` greater than `threshold`
    """
    return len([l for l in np.linalg.eigvalsh(C) if l >= threshold])

def all_order_selection(C):
    """Naive threshold model order estimator, but for all thresholds

    Parameters
    ----------
    C : ndarray of shape (N,N)
        Covariance matrix

    Returns
    -------
    list of floats
        List of thresholds
    list of ints
        List of order estimates
    """
    W = np.linalg.eigvalsh(C)
    thresholds = list(np.linspace(np.min(W), np.max(W), 100))
    orders = [np.sum(W >= threshold) for threshold in thresholds]
    return thresholds, orders

def min_description_length(C, m, p):
    """Minimum description length for a covariance matrix

    Parameters
    ----------
    C : ndarray of shape (N,N)
        Covariance matrix
    m : int
        Number of samples
    p : int
        Potential order of model

    Returns
    -------
    float
        MDL score
    """
    _, w, _ = np.linalg.svd(C)[::-1]
    w = np.abs(w[p:])
    n = C.shape[0]
    # abs() should do nothing here
    # but very small eigenvalues can end up negative through numerical silliness
    # which breaks the geometric mean
    MDL = - m * (n-p) * np.log(gmean(w) / np.mean(w)) + 0.5*p*(2*n-p)*np.log(m)
    return MDL

def mdl_order_selection(C, m):
    """Model order selection by minimizing MDL

    Parameters
    ----------
    C : ndarray of shape (N,N)
        Covariance matrix
    m : int
        Number of samples

    Returns
    -------
    int
        Model order
    """
    n = C.shape[0]
    ps = np.arange(1, n-1)
    costs = [min_description_length(C, m, p) for p in ps]
    idx = np.argmin(costs)

    return ps[idx]

def partition_recovery(C, k, laplacian=False):
    """Partition recovery algorithm

    Parameters
    ----------
    C : ndarray of shape (N,N)
        Covariance matrix
    k : int
        Number of partitions
    laplacian : bool
        Pass value True when a graph Laplacian is used. This will cluster based on the k smallest eigenvectors

    Returns
    -------
    list of ints
        Predicted node labeling
    """
    n = C.shape[0]
    if laplacian:
        _, V = scipy.linalg.eigh(C, eigvals=(0, k-1))
    else:
        _, V = scipy.linalg.eigh(C, eigvals=(n-k, n-1))

    kmeans = KMeans(n_clusters=k).fit(V)
    labels = kmeans.labels_

    return labels

def fraction_mislabeled_nodes(labels, labels_pred):
    """Return the fraction of mislabeled nodes across two labeling vectors

    Parameters
    ----------
    labels : list of ints
        Ground-truth node labeling
    labels_pred : list of ints
        Candidate node labeling

    Returns
    -------
    float
        Fraction of mislabeled nodes found (error rate)

    Notes
    -----
    Thanks Michael!
    """
    G1 = partition_indicator(labels)
    G2 = partition_indicator(labels_pred)

    # cost is minimized, overlap maximized
    cost_matrix = -G1.T.dot(G2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.A)
    cost = -cost_matrix[row_ind, col_ind].sum()

    return 1 - (cost / len(labels))

def overlap_score(labels, labels_pred):
    """Overlap score between two indicator matrices, as in [1]
    
    Parameters
    ----------
    labels : list of ints
        Ground-truth node labeling
    labels_pred : list of ints
        Candidate node labeling

    Returns
    -------
    float
        Fraction of mislabeled nodes found (error rate)

    References
    -----
    [1] Krzakala, Florent, et al. "Spectral redemption in clustering sparse networks." Proceedings of the National Academy of Sciences 110.52 (2013): 20935-20940.
    """
    raw_overlap = 1-fraction_mislabeled_nodes(labels, labels_pred)
    partition_true = np.array(labels).astype(int)
    partition_pred = np.array(labels_pred).astype(int)
    num_nodes = partition_pred.size
    num_groups = partition_true.max() + 1

    chance_level = 0.
    for i in range(num_groups):
        temp = np.sum(i == partition_true) / num_nodes
        if temp > chance_level:
            chance_level = temp

    score = (raw_overlap - chance_level) / (1 - chance_level)
    if score <= 0:
        score = 0

    return score
