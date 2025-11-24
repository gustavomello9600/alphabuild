import numpy as np
from scipy.sparse import csr_matrix, csgraph
from typing import Tuple, FrozenSet, List, Dict, Set
import sys

# Increase recursion depth for deep DFS on larger grids
sys.setrecursionlimit(10000)

from alphabuilder.src.logic.game_types import Coord

def grid_to_adjacency(grid: np.ndarray) -> Tuple[csr_matrix, List[Coord], Dict[Coord, int]]:
    """
    Converts a binary grid to a sparse adjacency matrix.
    Returns:
        - adj_matrix: scipy.sparse.csr_matrix
        - nodes: List of (r, c) coordinates corresponding to matrix indices
        - node_to_idx: Dict mapping (r, c) to matrix index
    """
    rows, cols = grid.shape
    nodes = []
    node_to_idx = {}
    
    # Identify all active nodes
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:
                idx = len(nodes)
                nodes.append((r, c))
                node_to_idx[(r, c)] = idx
                
    n_nodes = len(nodes)
    if n_nodes == 0:
        return csr_matrix((0, 0)), [], {}

    # Build edges (4-connectivity)
    data = []
    row_ind = []
    col_ind = []
    
    for idx, (r, c) in enumerate(nodes):
        # Check 4 neighbors
        neighbors = [
            (r-1, c), (r+1, c), 
            (r, c-1), (r, c+1)
        ]
        
        for nr, nc in neighbors:
            if (nr, nc) in node_to_idx:
                n_idx = node_to_idx[(nr, nc)]
                # Add edge (undirected, so we add both ways or just one and symmetrize)
                # Here we iterate all nodes, so we will encounter the other direction later.
                # But to be safe and explicit for CSR:
                row_ind.append(idx)
                col_ind.append(n_idx)
                data.append(1)
                
    adj_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n_nodes, n_nodes))
    return adj_matrix, nodes, node_to_idx

def check_global_connectivity(grid: np.ndarray, sources: Tuple[Coord, ...], targets: Tuple[Coord, ...]) -> bool:
    """
    Checks if all source and target nodes are part of the same connected component.
    Also implicitly checks if the structure itself is fully connected (single component),
    but strictly speaking, we care that loads connect to supports.
    """
    adj, nodes, node_to_idx = grid_to_adjacency(grid)
    
    if adj.shape[0] == 0:
        return False
        
    n_components, labels = csgraph.connected_components(adj, directed=False, return_labels=True)
    
    if n_components == 0:
        return False
        
    # Find which component the first valid source belongs to
    active_sources = [s for s in sources if s in node_to_idx]
    active_targets = [t for t in targets if t in node_to_idx]
    
    if not active_sources or not active_targets:
        return False
        
    # Get component ID of the first source
    first_source_idx = node_to_idx[active_sources[0]]
    target_component = labels[first_source_idx]
    
    # Check all other sources
    for s in active_sources[1:]:
        if labels[node_to_idx[s]] != target_component:
            return False
            
    # Check all targets
    for t in active_targets:
        if labels[node_to_idx[t]] != target_component:
            return False
            
    return True

def get_articulation_points(grid: np.ndarray) -> FrozenSet[Coord]:
    """
    Finds articulation points in the grid graph using a non-recursive DFS (to avoid recursion limit issues)
    or a robust recursive implementation if depth permits.
    Given the grid size (max ~3200 nodes), recursion might be risky but likely okay with sys.setrecursionlimit.
    We will use an iterative version of Tarjan's algorithm for robustness.
    """
    adj, nodes, node_to_idx = grid_to_adjacency(grid)
    n = adj.shape[0]
    
    if n == 0:
        return frozenset()
        
    # Adjacency list for faster traversal in Python
    # adj is CSR, so we can access neighbors efficiently
    indices = adj.indices
    indptr = adj.indptr
    
    def get_neighbors(u):
        return indices[indptr[u]:indptr[u+1]]

    # Tarjan's Algorithm variables
    ids = [-1] * n
    low = [-1] * n
    articulation_points_indices = set()
    
    timer = 0
    
    # Iterative DFS wrapper to handle all components
    visited = [False] * n
    
    for start_node in range(n):
        if ids[start_node] != -1:
            continue
            
        def dfs(u, p=-1):
            nonlocal timer
            visited[u] = True
            ids[u] = low[u] = timer
            timer += 1
            children = 0
            
            for v in get_neighbors(u):
                if v == p:
                    continue
                if ids[v] != -1:
                    # Back edge
                    low[u] = min(low[u], ids[v])
                else:
                    # Tree edge
                    children += 1
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if p != -1 and low[v] >= ids[u]:
                        articulation_points_indices.add(u)
            
            if p == -1 and children > 1:
                articulation_points_indices.add(u)

        dfs(start_node)

    return frozenset(nodes[i] for i in articulation_points_indices)
