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

class NOTEARS():
    def __init__(self, X, LAMBDA):
        self.X = X
        self.n = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.LAMBDA = LAMBDA

    ### OBJECTIVE FUNCTIONS
    def projection(self, W):
        R = self.X - self.X @ W

        return (1 / (2 * self.n) * (R ** 2).sum())

    def l1_pen(self, W):
        return np.abs(W).sum()
    
    def acyclic_restriction(self, W):
        W_squared = W * W

        return np.trace(expm(W_squared)) - self.dim

    def cost_function(self, W):
        return self.projection(W) + self.LAMBDA * self.l1_pen(W)

    def augmented_lagrangian(self, W, rho, alpha):
        cost = self.cost_function(W)
        h = self.acyclic_restriction(W)

        return (cost + (rho / 2) * h**2 + alpha * h)






class NOTEARSLagrangianProblem(Problem):

    def __init__(self, rho, alpha, X, LAMBDA):

        self.notears = NOTEARS(X, LAMBDA)
        self.dim = self.notears.dim
        self.rho = rho
        self.alpha = alpha

        upper_bounds = np.zeros((self.dim, self.dim))
        upper_bounds.fill(3)
        np.fill_diagonal(upper_bounds, 0)
        upper_bounds = upper_bounds.reshape(-1)
        lower_bounds = -1*upper_bounds

        super().__init__(
            n_var=self.notears.dim **2,
            n_obj=1,
            xl=lower_bounds,
            xu=upper_bounds
        )

    def _evaluate(self, W, out, *args, **kwargs):
        F = np.zeros(len(W))

        for i in range(len(W)):
            F[i] = self.notears.augmented_lagrangian(W[i].reshape(self.dim, self.dim), self.rho, self.alpha)

        out["F"] = F.reshape(-1, 1)



def notears_ga(X, LAMBDA, rho=1.0, alpha=0.0, rho_max=10e10, h_tol=1e-3, iter=100, pop_size=100, crossover_prob=0.7, mutation_prob=1.0, ga_gen_num=500, threshold=0.3):
    h = np.inf
    W_est = None

    for _ in range(iter):
        w_new, h_new = None, None
        while rho < rho_max:
            problem = NOTEARSLagrangianProblem(rho=rho, alpha=alpha, X=X, LAMBDA=LAMBDA)

            algorithm = GA(pop_size=pop_size,
                        crossover=SBX(
                            prob=crossover_prob,
                            eta=20
                        ),
                        mutation=PM(
                            prob=mutation_prob,
                            prob_var=1/(problem.dim**2),
                            eta=15
                        ),
                        eliminate_duplicates=True)
            
            res = minimize(
                problem,
                algorithm,
                termination=("n_gen", ga_gen_num),
                sampling=W_est if W_est is not None else np.zeros(problem.dim**2).reshape(problem.dim, problem.dim),
                seed=np.random.randint(100),
                verbose=False
            )

            notears = NOTEARS(X, LAMBDA)

            w_new = res.X.reshape(problem.dim, problem.dim)
            h_new = notears.acyclic_restriction(w_new)
            objective = notears.augmented_lagrangian(w_new, rho, alpha)

            print(f"Objetivo:       {objective:.8f}")
            print(f"h(W):           {h_new:.8f}")
            print(f"rho:            {rho:.4f}")
            print(f"alpha:          {alpha:.8f}")

            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

        W_est, h = w_new, h_new
        alpha += rho * h

        if h <= h_tol or rho >= rho_max:
            break

    W_est[np.abs(W_est) < threshold] = 0.0
    return W_est





class NOTEARS_multioo(ElementwiseProblem):
    def __init__(self, X, LAMBDA) :
        self.notears = NOTEARS(X, LAMBDA)
        self.dim = self.notears.dim

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

        f1 = self.notears.cost_function(x)
        f2 = self.notears.acyclic_restriction(x)

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


def notears_nsgaii(X, LAMBDA, pop_size=100, crossover_prob=0.6, mutation_prob=1.0, ga_gen_num=1500, seed=np.random.randint(100), threshold=0.3):
    problema = NOTEARS_multioo(X, LAMBDA)

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

    

