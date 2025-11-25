import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import time

class SIMPSolver:
    def __init__(self, nelx, nely, volfrac, penal, rmin, ft):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ft = ft # Filter type: 1=sensitivity, 2=density

    def solve(self):
        print(f"Starting SIMP optimization: {self.nelx}x{self.nely}, Vol: {self.volfrac}")
        start_time = time.time()
        
        # Max iterations
        max_loop = 200
        # Convergence criteria
        tol_x = 0.01
        
        # Initialize density
        x = self.volfrac * np.ones(self.nely * self.nelx, dtype=float)
        xPhys = x.copy()
        
        change = 1
        loop = 0
        
        metrics_history = []

        while change > tol_x and loop < max_loop:
            loop += 1
            
            # FE Analysis
            try:
                U = self.fe_analysis(xPhys)
            except Exception as e:
                print(f"FE Analysis failed at iteration {loop}: {e}")
                break
            
            # Objective function and sensitivity analysis
            # Extract displacements for all elements
            # edofMat is (nelx*nely, 8)
            # U is (ndof,)
            # U[edofMat] is (nelx*nely, 8)
            
            U_ele = U[self.edofMat]
            
            # Calculate compliance per element
            # ce = sum( (U_ele @ KE) * U_ele, axis=1 )
            # KE is (8,8)
            # U_ele @ KE is (nelx*nely, 8)
            # Element-wise multiplication with U_ele and sum
            
            ce = np.sum(np.dot(U_ele, self.KE) * U_ele, axis=1)
            
            c = np.sum((self.Emin + (xPhys ** self.penal) * (self.E0 - self.Emin)) * ce)
            dc = -self.penal * (xPhys ** (self.penal - 1)) * (self.E0 - self.Emin) * ce
            dv = np.ones(self.nely * self.nelx)
            
            # Filtering
            if self.ft == 1:
                dc = self.check(self.nelx, self.nely, self.rmin, x, dc)
            elif self.ft == 2:
                dc = self.check(self.nelx, self.nely, self.rmin, x, dc)
                dv = self.check(self.nelx, self.nely, self.rmin, x, dv)
            
            # Optimality Criteria
            try:
                xnew = self.oc(self.nelx, self.nely, x, self.volfrac, dc, dv, g=0.2)
            except Exception as e:
                print(f"OC update failed at iteration {loop}: {e}")
                break
            
            change = np.max(np.abs(xnew - x))
            x = xnew
            xPhys = x # For ft=1
            
            # Log metrics
            vol_current = np.sum(xPhys) / (self.nelx * self.nely)
            print(f" It.: {loop:4d} Obj.: {c:10.4f} Vol.: {vol_current:6.3f} ch.: {change:6.3f}")
            metrics_history.append({"iteration": loop, "compliance": float(c), "volume": float(vol_current), "change": float(change)})

        end_time = time.time()
        print(f"Optimization finished in {end_time - start_time:.2f}s")
        
        return xPhys, metrics_history

    def fe_analysis(self, x):
        # Material properties
        self.E0 = 1
        self.Emin = 1e-9
        nu = 0.3
        
        # Element stiffness matrix
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        self.KE = self.E0 / (1 - nu**2) * np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        
        # Node indices
        # Use Fortran order (column-major) to match standard topology optimization logic
        # Nodes are numbered down the columns.
        # Node (y, x) index = x * (nely + 1) + y
        # Grid dimensions: (nely+1) rows, (nelx+1) columns
        
        # Generate node grid
        # nodenrs[y, x] gives the node number
        nodenrs = np.arange(0, (self.nelx + 1) * (self.nely + 1)).reshape((self.nely + 1, self.nelx + 1), order='F')

        # Element DOFs
        # For each element at (y, x) (where y is row 0..nely-1, x is col 0..nelx-1):
        # Nodes are:
        # TL: (y, x)
        # TR: (y, x+1)
        # BR: (y+1, x+1)
        # BL: (y+1, x)
        
        # We need to flatten elements in the same order as 'x' (density array).
        # 'x' is usually flattened column-major in these codes?
        # Let's assume 'x' is flattened column-major (iterate y then x).
        
        edofMat = []
        for elx in range(self.nelx):
            for ely in range(self.nely):
                # Nodes
                n1 = nodenrs[ely, elx]     # TL
                n2 = nodenrs[ely, elx+1]   # TR
                n3 = nodenrs[ely+1, elx+1] # BR
                n4 = nodenrs[ely+1, elx]   # BL
                
                # DOFs: 2*node, 2*node+1
                # Order in KE: usually TL, TR, BR, BL (or similar, need to match KE definition)
                # Standard 99 line code KE order corresponds to:
                # [u1, v1, u2, v2, u3, v3, u4, v4]
                # Where 1=TL, 2=TR, 3=BR, 4=BL?
                # Let's verify standard code node mapping.
                # Standard code: 1: (-1,1), 2: (1,1), 3: (1,-1), 4: (-1,-1)
                # (-1, 1) is TL. (1, 1) is TR. (1, -1) is BR. (-1, -1) is BL.
                # So yes, TL, TR, BR, BL.
                
                edof = np.array([
                    2*n1, 2*n1+1, 
                    2*n2, 2*n2+1, 
                    2*n3, 2*n3+1, 
                    2*n4, 2*n4+1
                ])
                edofMat.append(edof)
        
        self.edofMat = np.array(edofMat)
        
        # Construct sparse matrix
        iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()
        
        # Stiffness values
        # x is flattened column-major to match the loop order above (elx, ely)
        # But wait, the loop above is: for elx: for ely.
        # This means ely changes fastest. This is column-major.
        # So x should be interpreted as column-major.
        
        sK = ((self.KE.flatten()[np.newaxis]).T * (self.Emin + (x ** self.penal) * (self.E0 - self.Emin))).flatten(order='F')
        
        # K matrix
        ndof = 2 * (self.nelx + 1) * (self.nely + 1)
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof))
        K = K.tocsc()
        
        # Load vector
        F = np.zeros(ndof)
        
        # Load at x=L (right edge), y=H/2 (middle)
        # Right edge is col index nelx.
        # Middle row is nely // 2.
        load_node = nodenrs[self.nely // 2, self.nelx]
        F[2 * load_node + 1] = -1.0 # Y direction (down)
        
        # Fixed DOFs
        # x=0 (left edge) -> col index 0
        fixed_nodes = nodenrs[:, 0]
        fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        
        # Solve
        all_dofs = np.arange(ndof)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        U = np.zeros(ndof)
        try:
            U[free_dofs] = spsolve(K[free_dofs, :][:, free_dofs], F[free_dofs])
        except Exception as e:
            # If singular, maybe try lsqr or add damping?
            # Or just raise
            raise e
            
        return U

    def check(self, nelx, nely, rmin, x, dc):
        dcn = np.zeros(nely * nelx)
        # x and dc are column-major flattened (ely changes fastest)
        
        # To make neighborhood search easier, reshape to 2D
        # order='F' because x was constructed/used in column-major loops
        x_grid = x.reshape((nely, nelx), order='F')
        dc_grid = dc.reshape((nely, nelx), order='F')
        dcn_grid = np.zeros_like(x_grid)
        
        for i in range(nelx):
            for j in range(nely):
                sum_val = 0.0
                val = 0.0
                
                # Search neighborhood
                i_min = max(i - int(np.floor(rmin)), 0)
                i_max = min(i + int(np.floor(rmin)) + 1, nelx)
                j_min = max(j - int(np.floor(rmin)), 0)
                j_max = min(j + int(np.floor(rmin)) + 1, nely)
                
                for ii in range(i_min, i_max):
                    for jj in range(j_min, j_max):
                        dist = np.sqrt((i - ii)**2 + (j - jj)**2)
                        if dist <= rmin:
                            fac = rmin - dist
                            sum_val += fac
                            val += fac * x_grid[jj, ii] * dc_grid[jj, ii]
                
                dcn_grid[j, i] = val / (x_grid[j, i] * sum_val)
                
        return dcn_grid.flatten(order='F')

    def oc(self, nelx, nely, x, volfrac, dc, dv, g):
        l1 = 0
        l2 = 1e9
        move = 0.2
        
        xnew = np.zeros(nelx * nely)
        
        # Safety break
        for _ in range(100):
            lmid = 0.5 * (l2 + l1)
            
            # Avoid division by zero
            if lmid < 1e-20:
                 lmid = 1e-20
                 
            # xnew = max(0, max(x-move, min(1, min(x+move, x * sqrt(-dc/dv/lmid)))))
            # Note: dc is negative, so -dc is positive. dv is positive.
            # If dc is positive (error), term is nan.
            
            term = x * np.sqrt(np.maximum(0, -dc) / dv / lmid)
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, term))))
            
            if np.sum(xnew) - volfrac * nelx * nely > 0:
                l1 = lmid
            else:
                l2 = lmid
            
            if (l1 + l2) < 1e-20:
                break
                
            if (l2 - l1) / (l1 + l2) < 1e-3:
                break
                
        return xnew

def main():
    # Parameters for 2x1 Beam
    # Mesh: 64x32
    nelx = 64
    nely = 32
    volfrac = 0.5
    penal = 3.0
    rmin = 1.5
    ft = 1 # Sensitivity filter
    
    solver = SIMPSolver(nelx, nely, volfrac, penal, rmin, ft)
    x_final, metrics = solver.solve()
    
    # Save metrics
    os.makedirs("alphabuilder/data/baselines", exist_ok=True)
    with open("alphabuilder/data/baselines/simp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Plot
    plt.figure(figsize=(10, 5))
    
    # x is column-major flattened. Reshape to (nely, nelx)
    grid = x_final.reshape((nely, nelx), order='F')
    
    # Plot
    # imshow origin='upper' by default.
    # grid[0,0] is top-left.
    # In our node grid, y=0 is row 0.
    # If y=0 is top, this is correct.
    # Usually in FEM y increases upwards.
    # If we defined y=0 as row 0 (top), and load at y=nely/2 (middle), and fixed at x=0.
    # This is consistent.
    
    plt.imshow(1 - grid, cmap='gray', interpolation='none')
    plt.title(f"SIMP Solution (Compliance: {metrics[-1]['compliance']:.4f})")
    plt.axis('off')
    plt.savefig("alphabuilder/data/baselines/simp_result.png")
    print("Results saved to alphabuilder/data/baselines/")

if __name__ == "__main__":
    main()

