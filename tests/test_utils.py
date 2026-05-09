import re
import numpy as np


LMAX = 5
MMAX_SIZE = 6


def _translate_c_expr_to_python(expr):
    expr = expr.replace("(FLOAT)", "")
    expr = expr.replace("sqrt", "np.sqrt")
    expr = expr.replace("MIN", "min")
    expr = expr.replace("MAX", "max")
    return expr


def _extract_p_assignments(cuda_source):
    pattern = re.compile(
        r"P\[(\d+)\]\[(\d+)\]\s*=\s*(.*?);",
        flags=re.DOTALL,
    )

    assignments = []
    for match in pattern.finditer(cuda_source):
        ell = int(match.group(1))
        m = int(match.group(2))
        expr = match.group(3)

        # Ignore the zero-initialization loop assignment:
        # P[ell][m] = (FLOAT)0.;
        if not match.group(1).isdigit() or not match.group(2).isdigit():
            continue

        expr = _translate_c_expr_to_python(expr)
        assignments.append((ell, m, expr))

    return assignments


def make_compute_pbar_all_lmax5_python_wrapper(cuda_source):
    assignments = _extract_p_assignments(cuda_source)

    def compute_pbar_all_lmax5_python_wrapper(ellmax, mu):
        ellmax_clipped = min(ellmax, LMAX)

        x = np.clip(mu, -1.0, 1.0)
        x2 = x * x
        s2 = max(0.0, 1.0 - x2)
        s = np.sqrt(s2)

        # Variables used by the hardcoded formulas
        x3 = x2 * x
        s3 = s2 * s
        x4 = x2 * x2
        s4 = s2 * s2
        x5 = x4 * x
        s5 = s4 * s

        P = np.zeros((MMAX_SIZE, MMAX_SIZE), dtype=np.float64)

        namespace = {"np": np, "x": x, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "s": s, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "min": min, "max": max}

        for ell, m, expr in assignments:
            if ell <= ellmax_clipped:
                P[ell, m] = eval(expr, {"__builtins__": {}}, namespace)

        return P

    return compute_pbar_all_lmax5_python_wrapper


cuda_source = r"""
__device__ inline void compute_pbar_all_lmax5(int ellmax, FLOAT mu, FLOAT P[MMAX_SIZE][MMAX_SIZE])                       \
{                                                                                                                          \
    ellmax = MIN(ellmax, ELLMAX);                                                                                          \
                                                                                                                           \
    FLOAT x  = clamp1(mu);                                                                                                 \
    FLOAT x2 = x * x;                                                                                                      \
    FLOAT s2 = MAX((FLOAT)0., (FLOAT)1. - x2);                                                                             \
    FLOAT s  = sqrt(s2);                                                                                                   \
                                                                                                                           \
    _Pragma("unroll")                                                                                                      \
    for (int ell = 0; ell < MMAX_SIZE; ell++) {                                                                            \
        _Pragma("unroll")                                                                                                  \
        for (int m = 0; m < MMAX_SIZE; m++) P[ell][m] = (FLOAT)0.;                                                        \
    }                                                                                                                      \
                                                                                                                           \
    P[0][0] = (FLOAT)1.;                                                                                                   \
    if (ellmax <= 0) return;                                                                                               \
                                                                                                                           \
    P[1][0] = x;                                                                                                           \
    P[1][1] = -(FLOAT)0.70710678118654752440 * s;                                                                          \
    if (ellmax <= 1) return;                                                                                               \
                                                                                                                           \
    FLOAT x3 = x2 * x;                                                                                                     \
                                                                                                                           \
    P[2][0] = ((FLOAT)0.5) * (((FLOAT)3.) * x2 - (FLOAT)1.);                                                              \
    P[2][1] = -(FLOAT)1.22474487139158904910 * x * s;                                                                      \
    P[2][2] =  (FLOAT)0.61237243569579452455 * s2;                                                                         \
    if (ellmax <= 2) return;                                                                                               \
                                                                                                                           \
    FLOAT s3 = s2 * s;                                                                                                     \
                                                                                                                           \
    P[3][0] =  ((FLOAT)0.5) * (((FLOAT)5.) * x3 - ((FLOAT)3.) * x);                                                       \
    P[3][1] = -(FLOAT)0.43301270189221932338 * ((((FLOAT)5.) * x2) - (FLOAT)1.) * s;                                     \
    P[3][2] =  (FLOAT)1.36930639376291527536 * x * s2;                                                                     \
    P[3][3] = -(FLOAT)0.55901699437494742410 * s3;                                                                         \
    if (ellmax <= 3) return;                                                                                               \
                                                                                                                           \
    FLOAT x4 = x2 * x2;                                                                                                    \
    FLOAT s4 = s2 * s2;                                                                                                    \
                                                                                                                           \
    P[4][0] =  ((FLOAT)0.125) * (((FLOAT)35.) * x4 - ((FLOAT)30.) * x2 + (FLOAT)3.);                                     \
    P[4][1] = -(FLOAT)0.55901699437494742410 * x * ((((FLOAT)7.) * x2) - (FLOAT)3.) * s;                                 \
    P[4][2] =  (FLOAT)0.39528470752104741743 * ((((FLOAT)7.) * x2) - (FLOAT)1.) * s2;                                    \
    P[4][3] = -(FLOAT)0.93541434669348534640 * x * s3;                                                                     \
    P[4][4] =  (FLOAT)0.52291251658379721705 * s4;                                                                         \
    if (ellmax <= 4) return;                                                                                               \
                                                                                                                           \
    FLOAT x5 = x4 * x;                                                                                                     \
    FLOAT s5 = s4 * s;                                                                                                     \
                                                                                                                           \
    P[5][0] =  ((FLOAT)0.125) * (((FLOAT)63.) * x5 - ((FLOAT)70.) * x3 + ((FLOAT)15.) * x);                              \
    P[5][1] = -(FLOAT)0.19882122822827110675 * ((((FLOAT)21.) * x4) - ((FLOAT)14.) * x2 + (FLOAT)1.) * s;               \
    P[5][2] =  (FLOAT)0.48412291827592711065 * x * ((((FLOAT)3.) * x2) - (FLOAT)1.) * s2;                                \
    P[5][3] = -(FLOAT)0.52291251658379721705 * ((((FLOAT)9.) * x2) - (FLOAT)1.) * s3;                                    \
    P[5][4] =  (FLOAT)1.16926793336685668103 * x * s4;                                                                     \
    P[5][5] = -(FLOAT)0.70156076002011400980 * s5;                                                                         \
}
"""

compute_pbar_all_lmax5_python_wrapper = make_compute_pbar_all_lmax5_python_wrapper(cuda_source)


import math
import numpy as np
from scipy.special import lpmv


LMAX = 5
MMAX_SIZE = 6


def reference_pbar_all_lmax5(ellmax, mu):
    ellmax = min(ellmax, LMAX)
    x = np.clip(mu, -1.0, 1.0)

    out = np.zeros((MMAX_SIZE, MMAX_SIZE), dtype=np.float64)

    for ell in range(ellmax + 1):
        for m in range(ell + 1):
            norm = math.sqrt(math.factorial(ell - m) / math.factorial(ell + m))
            out[ell, m] = norm * lpmv(m, ell, x)

    return out


def test_compute_pbar_all_lmax5():
    for ellmax in range(4):
        for mu in np.linspace(-1., 1., 10):
            expected = reference_pbar_all_lmax5(ellmax, mu)
            got = compute_pbar_all_lmax5_python_wrapper(ellmax, mu)
            assert np.allclose(got, expected)


if __name__ == '__main__':

    test_compute_pbar_all_lmax5()
    