import networkx as nx
import numpy as np
import pandas as pd

def generate_sem(n, adj_matrix, noise="gaussian", seed=np.random.randint(0, 100),
                 coef_range=(1, 2.5)):
    
    rng = np.random.default_rng(seed)
    
    adj_matrix = np.asarray(adj_matrix)
    
    # --------------------------------------------------
    # Validações
    # --------------------------------------------------
    if adj_matrix.ndim != 2:
        raise ValueError("adj_matrix deve ser uma matriz 2D.")
    
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("adj_matrix deve ser quadrada.")
    
    if not np.all(np.isin(adj_matrix, [0, 1])):
        raise ValueError("adj_matrix deve conter apenas 0 e 1.")
    
    d = adj_matrix.shape[0]
    
    # --------------------------------------------------
    # Verifica se o grafo é acíclico
    # --------------------------------------------------
    G = nx.DiGraph()
    
    for i in range(d):
        for j in range(d):
            if adj_matrix[i, j] == 1:
                G.add_edge(i, j)
    
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(
            "A matriz representa um grafo cíclico. "
            "Para um SEM causal, forneça um DAG."
        )
    
    # --------------------------------------------------
    # Matriz dos coeficientes
    # Mesma orientação da matriz de entrada:
    #
    # A[i,j] = 1  =>  Xi -> Xj
    # B[i,j] = coeficiente de Xi -> Xj
    # --------------------------------------------------
    coef_matrix = np.zeros((d, d))
    
    for i in range(d):
        for j in range(d):
            if adj_matrix[i, j] == 1:
                
                # Magnitude aleatória
                beta = rng.uniform(*coef_range)
                
                # Sinal aleatório
                beta *= rng.choice([-1, 1])
                
                coef_matrix[i, j] = beta
    
    # --------------------------------------------------
    # Geração dos dados na ordem topológica
    # --------------------------------------------------
    topological_order = list(nx.topological_sort(G))
    
    X = np.zeros((n, d))
    
    for j in topological_order:
        
        # Ruído da variável
        if noise == "gaussian":
            epsilon = rng.normal(size=n)
            
        elif noise == "laplace":
            epsilon = rng.laplace(size=n)
            
        else:
            raise ValueError(
                "noise deve ser 'gaussian' ou 'laplace'."
            )
        
        # Pais de Xj
        parents = np.where(adj_matrix[:, j] == 1)[0]
        
        # SEM linear
        X[:, j] = epsilon
        
        for i in parents:
            X[:, j] += coef_matrix[i, j] * X[:, i]
    
    # --------------------------------------------------
    # DataFrame
    # --------------------------------------------------
    columns = [f"X{i+1}" for i in range(d)]
    
    df = pd.DataFrame(X, columns=columns)
    
    return df, coef_matrix