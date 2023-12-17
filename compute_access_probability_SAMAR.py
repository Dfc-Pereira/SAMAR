import numpy as np
from itertools import permutations


def __idle(k, t, na, c):
    """
    Idle slot function for both f and h functions

    Args:
    - k (int): Number of current slot.
    - t (int): Number of consecutive slots each node were idle.
    - na (int): Number of active nodes.
    - c (int): Constants: [Desire slot (j), Desire number of active nodes (n_a), number of slots (N_S)].

    Returns:
    - f_i(k,t,n_a,c) and h_i(k,t,n_a,c)
    """
    q = 1 - 1 / (c[2] - t)
    f_i, h_i = __compute_f_and_h(k + 1, t + 1, na, c)
    f_i *= np.prod(q)
    h_i *= np.prod(q)
    return f_i, h_i


def __collision(k, t, na, c):
    """
    Collision slot function for both f and h functions

    Args:
    - k (int): Number of current slot.
    - t (int): Number of consecutive slots each node were idle.
    - na (int): Number of active nodes.
    - c (int): Constants: [Desire slot (j), Desire number of active nodes (n_a), number of slots (N_S)].

    Returns:
    - f_c(k,t,n_a,c) and h_c(k,t,n_a,c)
    """
    f_c = 0
    h_c = 0
    p = 1 / (c[2] - t)
    q = 1 - p
    for n_c in range(2, na + 1):
        aux = [1] * n_c + [0] * (na - n_c)
        n_c_colisions_i_index = np.array(list(set(permutations(aux, na))), dtype=bool)
        for i in n_c_colisions_i_index:
            j = ~ i
            t_aux = t + 1
            t_aux[i] = 0
            f_c_aux, h_c_aux = __compute_f_and_h(k + 1, t_aux, na, c)
            f_c_aux *= np.prod(p[i]) * np.prod(q[j])
            h_c_aux *= np.prod(p[i]) * np.prod(q[j])
            f_c += f_c_aux
            h_c += h_c_aux
    return f_c, h_c


def __success(k, t, na, c):
    """
    Success slot function for both f and h functions

    Args:
    - k (int): Number of current slot.
    - t (int): Number of consecutive slots each node were idle.
    - na (int): Number of active nodes.
    - c (int): Constants: [Desire slot (j), Desire number of active nodes (n_a), number of slots (N_S)].

    Returns:
    - f_s(k,t,n_a,c) and h_s(k,t,n_a,c)
    """
    f_s = 0
    h_s = 0
    p = 1 / (c[2] - t)
    q = 1 - p
    for i in range(na):
        j = np.arange(0, na) != i
        f_s_aux, h_s_aux = __compute_f_and_h(k + 1, t[j] + 1, na - 1, c)
        f_s_aux *= p[i] * np.prod(q[j])
        h_s_aux *= p[i] * np.prod(q[j])
        f_s += f_s_aux
        h_s += h_s_aux
    return f_s, h_s


def __compute_f_and_h(k, t, na, c):
    """
    Define f and h recursive functions

    Args:
    - k (int): Number of current slot.
    - t (int): Number of consecutive slots each node were idle.
    - na (int): Number of active nodes.
    - c (int): Constants: [Desire slot (j), Desire number of active nodes (n_a), number of slots (N_S)].

    Returns:
    - f(k,t,n_a,c) and h(k,t,n_a,c)
    """
    if k == c[0] and na != c[1]:
        return 0, 0  # (6a), (11a)
    elif k == c[0] and na == c[1]:
        return 1 / (c[2] - t[0]), 1  # (6b), (11b)
    elif k < c[0]:
        f_i, h_i = __idle(k, t, na, c)
        f_c, h_c = __collision(k, t, na, c)
        if na > c[1]:
            f_s, h_s = __success(k, t, na, c)
            return f_i + f_c + f_s, h_i + h_c + h_s  # (6c), (11c)
        return f_i + f_c, h_i + h_c  # (6d), (11d)


def __compute_tau_a(N_a, j):
    """
    Compute access probability at slot j with N_a active nodes, i.e., tau_a(N_a,j)  (5)

    Args:
    - N_a (int): Number of active nodes.
    - j (int): Number of slots

    Returns:
    - tau_a(N_a,j)
    """
    k = 1
    na = N
    t = np.zeros((na))
    c = [j, N_a, N_S]

    f, h = __compute_f_and_h(k, t, na, c)
    tau_a[N - N_a, j - 1] = f / h

    return tau_a[N - N_a, j - 1]


def __get_N_a_and_j_range():
    if N >= N_S:
        N_a_range = np.arange(N, N - N_S, -1)
        j_range = np.arange(1, N_S + 1)
    else:
        N_a_range = np.arange(N, 0, -1)
        j_range = np.arange(1, N_S + 1)

    return N_a_range, j_range


def get_tau_a():
    N_a_range, j_range = __get_N_a_and_j_range()
    for N_a in N_a_range:
        for j in j_range:
            if tau_a[N - N_a, j - 1] == 0:
                __compute_tau_a(N_a, j)
        j_range = j_range[1:]
    return tau_a


if __name__ == '__main__':
    N = 4  # define the number of initial nodes
    N_S = 4  # define the number of slots in the frame
    if N > N_S:
        tau_a = np.zeros((N_S, N_S))
    else:
        tau_a = np.zeros((N, N_S))
    print(get_tau_a())
