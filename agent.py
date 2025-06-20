from typeguard import typechecked
import cvxpy as cp
import numpy as np
import logging
import matplotlib.pyplot as plt

@typechecked
class Agent:
    
    def __init__(self,
                 id: int,
                 x0: np.ndarray, 
                 A: np.ndarray,
                 B: np.ndarray, 
                 N: int,
                 u_max: float
    ):
        self.id = id
        
        self.A = A
        self.B = B
        self.N = N 

        self.nx: int = A.shape[0]
        self.nu: int = B.shape[1]

        self.x0 = x0

        self.u = cp.Variable((self.nu, N-1), "input")
        self.x = cp.Variable((self.nx, N), "states")

        # Dual variables for dynamic evolution constraints
        self.lmbd_dyn = cp.Parameter((self.nx, N-1))
        # Dual variables for initial conditions
        self.lmbd_ini = cp.Parameter((self.nx, 1))
        # Dual variable for final state
        self.lmbd_fin_pos = cp.Parameter((self.nx, 1))
        self.lmbd_fin_neg = cp.Parameter((self.nx, 1))

        # Consensus ADMM variables
        self.x_f = cp.Parameter((self.nx, 1), "final_state")
        self.lmbd_adm = cp.Parameter((self.nx, 1))
        self.rho = cp.Parameter()

        self.iter_counter = 1


        self.u_max = u_max


    def set_local_dual_problem(self):

        state_evolution_error = self.x[:, 1:] - (self.A @ self.x[:, :-1] + self.B @ self.u)

        # Subproblem for the agent
        cost = 0
        # Minimize state trajectory magnitude
        cost += cp.sum_squares(self.x)
        # Minimize input trajectory magnitude
        cost += cp.sum_squares(self.u)
        # Lagrange function
        lagrange = 0
        # Add original cost
        lagrange += cost
        # Add initial condition (maybe this can be moved)
        lagrange += cp.sum(cp.multiply(self.lmbd_ini, (self.x[:, 0:1] - self.x0)))
        # Add dynamic evolution constraint
        lagrange += cp.sum(cp.multiply(self.lmbd_dyn, state_evolution_error))
        # Add final state constraint
        lagrange += self.lmbd_fin_pos.T @ self.x[:, self.N-1:self.N]
        lagrange += self.lmbd_fin_neg.T @ self.x[:, self.N-1:self.N]
        
        cost = cp.Minimize(lagrange)

        subproblem_constraints = [
            # self.u <= self.u_max, 
            # self.u >= -self.u_max
            cp.sum_squares(self.u) <= 5*self.u_max**2
            ]
       
        self.local_problem = cp.Problem(cost, subproblem_constraints)

    
    def set_local_admm_problem(self):

        # # Subproblem for the agent
        # cost = 0
        # # Minimize state trajectory magnitude
        # cost += cp.sum_squares(self.x)
        # # Minimize input trajectory magnitude
        # cost += cp.sum_squares(self.u)
        # # Lagrange function
        # aug_lagrange = 0
        # # Add original cost
        # aug_lagrange += cost
        # # Add final state constraint
        # aug_lagrange += self.lmbd_adm.T @ (self.x[:, self.N-1:self.N] - self.x_f)
        # print(f"Error shape: {(self.rho/2 * cp.square(self.x[0, self.N-1:self.N] - 7)).shape}")
        # tmp = self.rho * self.x[0, self.N-1:self.N]
        # aug_lagrange += self.rho/2 * tmp
        
        # cost = cp.Minimize(aug_lagrange)

        # subproblem_constraints = [
        #     self.x[:, 1:] == (self.A @ self.x[:, :-1] + self.B @ self.u),
        #     self.x[:, 0:1] == self.x0,
        #     self.u <= self.u_max, 
        #     self.u >= -self.u_max
        #     ]
       
        # self.local_admm_problem = cp.Problem(cost, subproblem_constraints)
        pass


    def solve_local(self, lmbd_ini, lmbd_dyn, lmbd_fin_pos, lmbd_fin_neg):
        
        self.lmbd_ini.value = lmbd_ini
        self.lmbd_dyn.value = lmbd_dyn
        self.lmbd_fin_pos.value = lmbd_fin_pos
        self.lmbd_fin_neg.value = lmbd_fin_neg
        # print(f"Local Optimization of Agent {self.id} has started")
        result = self.local_problem.solve(verbose=False, solver=cp.SCS)
        # Gurobi for QCQP
        # result = self.local_problem.solve(verbose=False, solver=cp.GUROBI)
        # print(f"Solver status: {self.local_problem.status}")
        # print(f"Solver optimal cost: {self.local_problem.value}")
        # print(f"Solver result: {result}")
        x = self.x.value
        u = self.u.value

        # Print the left and right hand side of the constraint
        lhs = np.sum(u**2)
        rhs = 5 * self.u_max**2
        print(f"Constraint check: sum_squares(u) = {lhs}, 5*u_max^2 = {rhs}")
        # print(self.x.value)
        # print(self.u.value)

        # Plot and save the x trajectory
        if self.iter_counter % 1000 == 0:
            self.iter_counter = 1
            plt.figure()
            for idx in range(self.nx):
                plt.plot(range(self.N), self.x.value[idx, :], label=f'x{idx}')
            plt.xlabel('Time step')
            plt.ylabel('State value')
            plt.title('State Trajectory')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'x_trajectory_agent_{self.id}.png')
        else: 
            self.iter_counter += 1
        
        return x, u
    

    def solve_admm_local(self, lmbd_adm, x_f, rho):

        # Subproblem for the agent
        cost = 0
        # Minimize state trajectory magnitude
        cost += cp.sum_squares(self.x)
        # Minimize input trajectory magnitude
        cost += cp.sum_squares(self.u)
        # Lagrange function
        aug_lagrange = 0
        # Add original cost
        aug_lagrange += cost
        # Add final state constraint
        aug_lagrange += lmbd_adm.T @ (self.x[:, self.N-1:self.N] - self.x_f)
        aug_lagrange += rho/2 * cp.sum_squares(self.x[:, self.N-1:self.N] - x_f)
        
        cost = cp.Minimize(aug_lagrange)

        subproblem_constraints = [
            self.x[:, 1:] == (self.A @ self.x[:, :-1] + self.B @ self.u),
            self.x[:, 0:1] == self.x0,
            # self.u <= self.u_max, 
            # self.u >= -self.u_max
            cp.sum_squares(self.u) <= 1/5*self.u_max**2
            ]
       
        self.local_admm_problem = cp.Problem(cost, subproblem_constraints)
        


        self.lmbd_adm.value = lmbd_adm
        self.x_f.value = x_f
        self.rho.value = rho
        # print(f"Local Optimization of Agent {self.id} has started")
        self.local_admm_problem.solve(verbose=False, solver=cp.SCS)
        # Gurobi for QCQP
        # result = self.local_problem.solve(verbose=False, solver=cp.GUROBI)
        # print(f"Solver status: {self.local_admm_problem.status}")
        # print(f"Solver optimal cost: {self.local_admm_problem.value}")
        x = self.x.value
        u = self.u.value
        # print(self.x.value)
        # print(self.u.value)
        # Print the left and right hand side of the constraint
        lhs = np.sum(u**2)
        rhs = self.u_max**2
        print(f"Constraint check: sum_squares(u) = {lhs}, 5*u_max^2 = {rhs}")

        # Plot and save the x trajectory
        if self.iter_counter % 1000 == 0:
            self.iter_counter = 1
            plt.figure()
            for idx in range(self.nx):
                plt.plot(range(self.N), self.x.value[idx, :], label=f'x{idx}')
            plt.xlabel('Time step')
            plt.ylabel('State value')
            plt.title('State Trajectory')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'x_trajectory_agent_{self.id}.png')
        else: 
            self.iter_counter += 1
        
        return x, u

