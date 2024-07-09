import warnings
import numpy as np

def system_info(a, b, c):

    ab_contr = np.hstack((b, a @ b, a @ a @ b))
    ab_row_rank = np.linalg.matrix_rank(ab_contr)

    ac_obs = np.hstack((c.T, (c @ a).T, (c @ a @ a).T)).T
    ac_column_rank = np.linalg.matrix_rank(ac_obs)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fi_a_vect = [np.linalg.norm(np.linalg.matrix_power(a, i)) /
                     np.max(np.abs(np.linalg.eigvals(a)))**i
                     for i in range(1, 1000)]

    print('Eigenvalues: ', np.linalg.eigvals(a))
    print('Maximum Eigenvalue: ', np.max(np.abs(np.linalg.eigvals(a))))
    print('Singular Values: ', np.linalg.eigvals(a @ a.T))
    print('Ab controllability matrix rank: ', ab_row_rank)
    print('Ac observability matrix rank: ', ac_column_rank)
    print('Sup(fi(A)): ', max(fi_a_vect))


def compute_bar_a(A, B, C, K):
    top_left = A - (K[0]+K[2]) * B @ C
    top_mid = B * K[1]
    top_right = -B * K[2]
    mid_left = -C
    mid_mid = np.array([[1]])
    mid_right = np.array([[0]])

    bar_A = np.block([
    [top_left, top_mid, top_right],
    [mid_left, mid_mid, mid_right],
    [mid_left, mid_right, mid_right]
    ])

    return bar_A


def spectr(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rho = max(np.abs(np.linalg.eigvals(a)))
        fi_a_vect = []
        for i in range(1, 1000):
            try:
                a_power_i = np.linalg.matrix_power(a, i)
                norm_a_power_i = np.linalg.norm(a_power_i, ord=2)
                fi_a_vect.append(norm_a_power_i / (rho ** i))
            except Exception as e:
                print(f"Error at i={i}: {e}")
                break
        return max(fi_a_vect, default=float('nan'))
