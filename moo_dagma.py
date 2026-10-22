from scipy.linalg import expm
import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.visualization.scatter import Scatter
import matplotlib.pyplot as plt

class DAGMA():
    def __init__(self, X, LAMBDA):
        self.X = X
        self.n = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.LAMBDA = LAMBDA

    ### OBJECTIVE FUNCTIONS
    def least_squares(self, W):
        R = self.X - self.X @ W

        return (1 / (2 * self.n) * (R ** 2).sum())

    def l1_pen(self, W):
        return np.abs(W).sum()
    
    def acyclic_restriction(self, W, s=1):
        W_squared = W * W
        M = s * np.identity(self.dim) - W_squared
        h = - np.log(np.linalg.det(M)) + self.dim * np.log(s)

        return h

    def cost_function(self, W):
        return self.least_squares(W) + self.LAMBDA * self.l1_pen(W)





class DAGMA_multioo(ElementwiseProblem):
    def __init__(self, X, LAMBDA) :
        self.dagma = DAGMA(X, LAMBDA)
        self.dim = self.dagma.dim

        upper_bounds = np.zeros((self.dim, self.dim))
        upper_bounds.fill(3)
        np.fill_diagonal(upper_bounds, 0)
        upper_bounds = upper_bounds.reshape(-1)
        lower_bounds = -1*upper_bounds

        super().__init__(
        n_var = self.dim**2,
        n_obj = 2,
        n_ieq_constr = 0,
        xl = lower_bounds,
        xu = upper_bounds
        )
        
    def _evaluate(self, x, out, *args, **kwargs):
        x = x.reshape(self.dim, self.dim)

        f1 = self.dagma.cost_function(x)
        f2 = self.dagma.acyclic_restriction(x)

        out["F"] = [ f1, f2 ]


def plot_fronteira_pareto(res):
    axis_labels = [
    r"$f_1(W)=\|X-XW\|_F^2+\lambda\|W\|_1$",
    r"$f_2(W)=\operatorname{tr}(e^{W\circ W})-d$"
]

    plot = Scatter(labels=axis_labels)

    plot.add(
        res.F,
        facecolor="none",
        edgecolor="red"
    )

    plot.do()
    plt.tight_layout()
    plt.show()


def dagma_nsgaii(X, LAMBDA, pop_size=100, crossover_prob=0.6, mutation_prob=1.0, ga_gen_num=1500, seed=np.random.randint(100), threshold=0.3):
    problema = DAGMA_multioo(X, LAMBDA)

    algoritmo = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(
                prob=crossover_prob,
                eta=15
            ),
            mutation=PM(
                prob=mutation_prob,
                prob_var=1/(problema.dim**2),
                eta=20
            ),
            eliminate_duplicates=True)

    res = minimize(problema, algoritmo, ('n_gen', ga_gen_num), verbose=False, seed=seed, save_history=True)

    def decisao(res):
        F = np.asarray(res.F, dtype=float)

        scale = np.ptp(F, axis=0)
        scale[scale == 0] = 1.0

        F_norm = F / scale
        distances = np.linalg.norm(F_norm, axis=1)

        idx = np.argmin(distances)

        return (res.X[idx], res.F[idx])

    plot_fronteira_pareto(res)

    W_final, dec_objs = decisao(res)
    W_final[np.abs(W_final) < threshold] = 0.0

    return res, W_final, dec_objs

    

