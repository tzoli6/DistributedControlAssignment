from agent import Agent
import numpy as np
import cvxpy as cp
import seaborn as sns
import os
import matplotlib.pyplot as plt


A1= np.array([[0.626, -0.6975], [0.8719, 0.626]])
B1= np.array([[-0.0748], [0.1744]])
x01= np.array([[-2], [-0.5]])


A2 = np.array([[0.626, -0.6975], [0.8719, 0.626]])
B2 = np.array([[-0.0748], [0.1744]])
x02 = np.array([[-1], [1]])

A3 = np.array([[0.626, -0.6975], [0.8719, 0.626]])
B3 = np.array([[-0.0748], [0.1744]])
x03 = np.array([[2], [0]])

A4 = np.array([[0.626, -0.6975], [0.8719, 0.626]])
B4 = np.array([[-0.0748], [0.1744]])
x04 = np.array([[3], [-0.5]])

N=20
umax=7

nx = 2
nu = 1

x_i0 = [x01, x02, x03, x04]
A_i = [A1, A2, A3, A4]
B_i = [B1, B2, B3, B4]

class Coordinator:

    def __init__(self,
                 n_agents: int,
                 x_i0: list[np.ndarray],
                 A_i: list[np.ndarray], 
                 B_i: list[np.ndarray],
                 N: int,
                 u_max: float,
                 ):
        
        self.n_agents = n_agents

        self.B_i = B_i
        self.A_i = A_i
        self.x_i0 = x_i0

        # Initialize agents
        self.agents: list[Agent] = []
        for i in range(n_agents):
            agent = Agent(
                i,
                np.array(x_i0[i]),
                np.array(A_i[i]),
                np.array(B_i[i]),
                N,
                u_max,
            )
            self.agents.append(agent)

        # Dual variables for dynamic evolution constraints
        self.lmbd_dyn = np.random.randn(n_agents, nx, N-1)
        self.momentum_dyn = self.lmbd_dyn.copy()
        # Dual variables for initial conditions
        self.lmbd_ini = np.ones((n_agents, nx, 1))
        self.momentum_ini = np.ones((n_agents, nx, 1))
        # Dual variable for final state
        self.lmbd_fin = np.ones((n_agents, nx, 1))
        self.lmbd_fin_local = np.ones((n_agents, n_agents, nx, 1))
        self.momentum_fin = np.ones((n_agents, nx, 1))

        # COnsensus ADMM variables
        self.x_f = np.zeros((nx, 1))
        self.lmbd_adm = np.ones((self.n_agents, nx, 1))

        self.beta = 1
        self.beta_f = 1
        
        self.eta = 0.0005
        self.eta_f = 0.0005

        self.consensus_matrix = np.array([
            [0.75, 0.25, 0,    0],
            [0.25, 0.5,  0.25, 0],
            [0,    0.25, 0.5,  0.25],
            [0,    0,    0.25, 0.75],
        ])

        self.iter_counter = 1
        self.n_iter = 500

        self.x = [0]*self.n_agents
        self.u = [0]*self.n_agents

        # Initialize agents
        for agent in self.agents:
            agent.set_local_dual_problem()
            agent.set_local_admm_problem()

        self.subgradient_history = {
            "initial": [],
            "dynamic": [],
            "final": []
        }

        # self.step_size_analysis()
        # self.consensus_iteration_number_analysis()
        # self.rho_analysis()
        for _ in range(self.n_iter):
            self.nesterov_sugradient_step()

        # Save x and u as numpy arrays after the iterations
        self.x = np.array(self.x)
        self.u = np.array(self.u)
        # Save to file
        np.save("x_history.npy", self.x)
        np.save("u_history.npy", self.u)

        # Print final states
        for x_i in self.x:
            print(f"{x_i[:, -1]}\n")


    def step_size_analysis(self):
        params = np.linspace(0.1, 1, 5)
        # Store subgradient histories for each param
        all_histories = []

        for param in params:
            # Reset subgradient history for each param
            self.subgradient_history = {
            "initial": [],
            "dynamic": [],
            "final": []
            }

            for _ in range(self.n_iter): 
                self.subgradient_step(param)

            self.iter_counter = 1
            self.lmbd_dyn = np.random.randn(self.n_agents, nx, N-1)
            self.lmbd_ini = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin_local = np.ones((self.n_agents, self.n_agents, nx, 1))

            # Store a copy of the history for this param
            all_histories.append({
            "param": param,
            "history": {
                "initial": [h.copy() for h in self.subgradient_history["initial"]],
                "dynamic": [h.copy() for h in self.subgradient_history["dynamic"]],
                "final": [h.copy() for h in self.subgradient_history["final"]],
            }
            })

        # Plot only for agent 1
        agent_idx = 0
        subgrad_types = ["initial", "dynamic", "final"]
        n_types = len(subgrad_types)
        n_params = len(params)

        fig_height = 3.5 * n_types
        fig, axes = plt.subplots(n_types, 1, figsize=(10, fig_height), sharex='col')
        colors = sns.color_palette("viridis", n_params)
        for type_idx, subgrad_type in enumerate(subgrad_types):
            ax = axes[type_idx]
            # Remove top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for p_idx, param_result in enumerate(all_histories):
                history = param_result["history"][subgrad_type]
                norms = [np.linalg.norm(h[agent_idx]) for h in history]
                ax.semilogy(norms, label=f"a={param_result['param']}", color=colors[p_idx], linewidth=2.2, alpha=0.95)
                ax.set_title(f"Agent {agent_idx+1} - {subgrad_type.capitalize()}", fontsize=16)
                ax.set_ylabel("Subgradient Norm", fontsize=16)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.legend(fontsize=13)
        axes[-1].set_xlabel("Iteration", fontsize=16)
        plt.tight_layout(pad=2.0)
        os.makedirs("standard_subgradient", exist_ok=True)
        plt.savefig("standard_subgradient/step_size_analysis_fixed.png", bbox_inches="tight", dpi=200)
        plt.close()


    def consensus_iteration_number_analysis(self):
        phis = np.arange(1, 5, 1)
        # Store subgradient histories for each param
        all_histories = []

        for phi in phis:
            # Reset subgradient history for each param
            self.subgradient_history = {
            "initial": [],
            "dynamic": [],
            "final": []
            }
            # Run a fixed number of iterations for each param
            for _ in range(500):  # Adjust as needed
                self.consensus_subgradient_step(0.7, phi)

            self.iter_counter = 1
            self.lmbd_dyn = np.ones((self.n_agents, nx, N-1))
            self.lmbd_ini = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin_local = np.ones((self.n_agents, self.n_agents, nx, 1))

            # Store a copy of the history for this param
            all_histories.append({
            "param": phi,
            "history": {
                "initial": [h.copy() for h in self.subgradient_history["initial"]],
                "dynamic": [h.copy() for h in self.subgradient_history["dynamic"]],
                "final": [h.copy() for h in self.subgradient_history["final"]],
            }
            })

        # Plot only for agent 1
        agent_idx = 0
        subgrad_types = ["initial", "dynamic", "final"]
        n_types = len(subgrad_types)
        n_params = len(phis)

        fig_height = 3.5 * n_types
        fig, axes = plt.subplots(n_types, 1, figsize=(10, fig_height), sharex='col')
        colors = sns.color_palette("viridis", n_params)
        for type_idx, subgrad_type in enumerate(subgrad_types):
            ax = axes[type_idx]
            # Remove top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for p_idx, param_result in enumerate(all_histories):
                history = param_result["history"][subgrad_type]
                norms = [np.linalg.norm(h[agent_idx]) for h in history]
                ax.semilogy(norms, label=f"param={param_result['param']}", color=colors[p_idx], linewidth=2.2, alpha=0.95)
                ax.set_title(f"Agent {agent_idx+1} - {subgrad_type.capitalize()}", fontsize=16)
                ax.set_ylabel("Subgradient Norm", fontsize=16)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.legend(fontsize=13)
        axes[-1].set_xlabel("Iteration", fontsize=16)
        plt.tight_layout(pad=2.0)
        os.makedirs("consensus_standard_subgradient", exist_ok=True)
        plt.savefig("consensus_standard_subgradient/consensus_iteration_number_analysis.png", bbox_inches="tight", dpi=200)
        plt.close()


    def rho_analysis(self):

        rhos = np.linspace(1, 20, 5)
        # Store subgradient histories for each param
        all_histories = []

        for rho in rhos:
            # Reset subgradient history for each param
            self.subgradient_history = {
            "initial": [],
            "dynamic": [],
            "final": []
            }
            # Run a fixed number of iterations for each param
            for _ in range(500):  # Adjust as needed
                self.consensus_admm(rho)

            self.iter_counter = 1
            self.lmbd_dyn = np.ones((self.n_agents, nx, N-1))
            self.lmbd_ini = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin = np.ones((self.n_agents, nx, 1))
            self.lmbd_fin_local = np.ones((self.n_agents, self.n_agents, nx, 1))
            self.x_f = np.zeros((nx, 1))
            self.lmbd_adm = np.ones((self.n_agents, nx, 1))

            # Store a copy of the history for this param
            all_histories.append({
            "param": rho,
            "history": {
                "initial": [h.copy() for h in self.subgradient_history["initial"]],
                "dynamic": [h.copy() for h in self.subgradient_history["dynamic"]],
                "final": [h.copy() for h in self.subgradient_history["final"]],
            }
            })

        # Plot only the final subgradient for all agents in different subplots
        subgrad_type = "final"
        n_agents = self.n_agents
        n_params = len(rhos)

        fig_height = 3.5 * n_agents
        fig, axes = plt.subplots(n_agents, 1, figsize=(10, fig_height), sharex=True)
        if n_agents == 1:
            axes = [axes]
        colors = sns.color_palette("viridis", n_params)
        for agent_idx in range(n_agents):
            ax = axes[agent_idx]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for p_idx, param_result in enumerate(all_histories):
                history = param_result["history"][subgrad_type]
                norms = [np.linalg.norm(h[agent_idx]) for h in history]
                param_str = f"{param_result['param']:.2f}"
                ax.semilogy(norms, label=f"rho={param_str}", color=colors[p_idx], linewidth=2.2, alpha=0.95)
                ax.set_title(f"Agent {agent_idx+1} - Final", fontsize=16)
                ax.set_ylabel("Subgradient Norm", fontsize=16)
                ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
                ax.legend(fontsize=13)
        axes[-1].set_xlabel("Iteration", fontsize=16)
        plt.tight_layout(pad=2.0)
        os.makedirs("consensus_admm", exist_ok=True)
        plt.savefig("consensus_admm/rho_analysis.png", bbox_inches="tight", dpi=200)
        plt.close()


    def subgradient_step(self, param):
        # Initialize accumulators for subgradients
        initial_cond_subg_total = np.zeros((self.n_agents, nx, 1))
        dynamic_evol_subg_total = np.zeros((self.n_agents, nx, N-1))
        final_state_subg_total = np.zeros((self.n_agents, nx, 1))

        for i, agent in enumerate(self.agents):
            lambda_pos = self.lmbd_fin[i, :, :]
            # Circular behaviour
            if i == 0:
                lambda_neg = self.lmbd_fin[-1, :, :]
            else:
                lambda_neg = self.lmbd_fin[i-1, :, :]

            # Perform local updates
            x, u = agent.solve_local(self.lmbd_ini[i, :, :], self.lmbd_dyn[i, :, :], lambda_pos, lambda_neg)

            # Save solution
            self.x[i] = x
            self.u[i] = u

        for i, _ in enumerate(self.agents):
            x = self.x[i] 
            u = self.u[i] 

            initial_cond_subg = x[:, 0:1] - self.x_i0[i]
            dynamic_evol_subg = x[:, 1:] - (self.A_i[i] @ x[:, :-1] + self.B_i[i] @ u)
            final_state_neg_subg = -x[:, N-1:N]
            final_state_pos_subg = x[:, N-1:N]

            # Save subgradients
            initial_cond_subg_total[i] = initial_cond_subg
            dynamic_evol_subg_total[i] = dynamic_evol_subg
            final_state_subg_total[i] += final_state_pos_subg
            if i == 0: final_state_subg_total[-1] += final_state_neg_subg
            else: final_state_subg_total[i-1] += final_state_neg_subg


            # Update dual variables
            alpha = param #self.alpha_sq(self.iter_counter, param)
            self.lmbd_ini[i, :, :] += alpha * initial_cond_subg
            self.lmbd_dyn[i, :, :] += alpha * dynamic_evol_subg
            self.lmbd_fin[i, :, :] += alpha * final_state_pos_subg
            if i == 0: self.lmbd_fin[-1, :, :] += alpha * final_state_neg_subg
            else: self.lmbd_fin[i-1, :, :] += alpha * final_state_neg_subg

        
        self.subgradient_history["initial"].append(initial_cond_subg_total.copy())
        self.subgradient_history["dynamic"].append(dynamic_evol_subg_total.copy())
        self.subgradient_history["final"].append(final_state_subg_total.copy())

        if self.iter_counter % self.n_iter == 0:
            self.iter_counter = 1
            #self.plot_convergence('standard_subgradient', 'subgradient_history.png')
            
        else:
            self.iter_counter += 1


    def nesterov_sugradient_step(self):
        # Initialize accumulators for subgradients
        initial_cond_subg_total = np.zeros((self.n_agents, nx, 1))
        dynamic_evol_subg_total = np.zeros((self.n_agents, nx, N-1))
        final_state_subg_total = np.zeros((self.n_agents, nx, 1))

        self.eta = self.alpha_sq(self.iter_counter, 0.7)
        self.eta_f = self.alpha_sq(self.iter_counter, 0.7)


        for i, agent in enumerate(self.agents):
            # Circular behaviour
            if i == 0:
                lambda_neg = self.lmbd_fin[-1, :, :]
                momentum_fin_neg = self.momentum_fin[-1, :, :]
            else:
                lambda_neg = self.lmbd_fin[i-1, :, :]
                momentum_fin_neg = self.momentum_fin[i-1, :, :]
            lambda_pos = self.lmbd_fin[i, :, :]
            momentum_fin_pos = self.momentum_fin[i, :, :]

            lmbd_ini = (1-self.beta) * self.lmbd_ini[i, :, :] + self.beta * self.momentum_ini[i, :, :]
            lmbd_dyn = (1-self.beta) * self.lmbd_dyn[i, :, :] + self.beta * self.momentum_dyn[i, :, :]

            lambda_pos = (1-self.beta_f) * lambda_pos + self.beta_f * momentum_fin_pos
            lambda_neg = (1-self.beta_f) * lambda_neg + self.beta_f * momentum_fin_neg

            x, u = agent.solve_local(lmbd_ini, lmbd_dyn, lambda_pos, lambda_neg)
                        
            self.x[i] = x
            self.u[i] = u
        
        
        for i, _ in enumerate(self.agents):
            x = self.x[i] 
            u = self.u[i] 


            # Calculate dual variable update
            if i == 0:
                lambda_neg = self.lmbd_fin[-1, :, :]
                momentum_fin_neg = self.momentum_fin[-1, :, :]
            else:
                lambda_neg = self.lmbd_fin[i-1, :, :]
                momentum_fin_neg = self.momentum_fin[i-1, :, :]
            lambda_pos = self.lmbd_fin[i, :, :]
            momentum_fin_pos = self.momentum_fin[i, :, :]


            lmbd_dyn = (1-self.beta) * self.lmbd_dyn[i, :, :] + self.beta * self.momentum_dyn[i, :, :]
            lambda_pos = (1-self.beta_f) * lambda_pos + self.beta_f * momentum_fin_pos
            lambda_neg = (1-self.beta_f) * lambda_neg + self.beta_f * momentum_fin_neg
            lmbd_ini = (1-self.beta) * self.lmbd_ini[i, :, :] + self.beta * self.momentum_ini[i, :, :]
            

            initial_cond_subg = x[:, 0:1] - self.x_i0[i]
            dynamic_evol_subg = x[:, 1:] - (self.A_i[i] @ x[:, :-1] + self.B_i[i] @ u)
            final_state_neg_subg = -x[:, N-1:N]
            final_state_pos_subg = x[:, N-1:N]

            # Accumulate subgradients
            initial_cond_subg_total[i] = initial_cond_subg
            dynamic_evol_subg_total[i] = dynamic_evol_subg
            final_state_subg_total[i] += final_state_pos_subg
            # Circular behaviour
            if i == 0: final_state_subg_total[-1] += final_state_neg_subg
            else: final_state_subg_total[i-1] += final_state_neg_subg

            # Calculate momentum step 1
            self.momentum_ini[i, :, :] = (1 - 1/self.beta) * self.lmbd_ini[i, :, :]
            self.momentum_dyn[i, :, :] = (1 - 1/self.beta) * self.lmbd_dyn[i, :, :]
            self.momentum_fin[i, :, :] = (1 - 1/self.beta) * self.lmbd_fin[i, :, :]
            if i == 0:  self.momentum_fin[-1, :, :]  = (1 - 1/self.beta) * self.lmbd_fin[-1, :, :]
            else:       self.momentum_fin[i-1, :, :] = (1 - 1/self.beta) * self.lmbd_fin[i-1, :, :]


            self.lmbd_ini[i, :, :] = lmbd_ini + self.eta * initial_cond_subg
            self.lmbd_dyn[i, :, :] = lmbd_dyn + self.eta * dynamic_evol_subg
            self.lmbd_fin[i, :, :] = lambda_pos + self.eta_f * final_state_pos_subg
            if i == 0:  self.lmbd_fin[-1, :, :]  = lambda_neg + self.eta_f * final_state_neg_subg
            else:       self.lmbd_fin[i-1, :, :] = lambda_neg + self.eta_f * final_state_neg_subg

            # Calculate momentum step 2
            self.momentum_ini[i, :, :] += 1/self.beta * self.lmbd_ini[i, :, :]
            self.momentum_dyn[i, :, :] += 1/self.beta * self.lmbd_dyn[i, :, :]
            self.momentum_fin[i, :, :] += 1/self.beta_f * self.lmbd_fin[i, :, :]
            if i == 0:  self.momentum_fin[-1, :, :]  += 1/self.beta_f * self.lmbd_fin[-1, :, :]
            else:       self.momentum_fin[i-1, :, :] += 1/self.beta_f * self.lmbd_fin[i-1, :, :]

        self.subgradient_history["initial"].append(initial_cond_subg_total.copy())
        self.subgradient_history["dynamic"].append(dynamic_evol_subg_total.copy())
        self.subgradient_history["final"].append(final_state_subg_total.copy())

        if self.iter_counter % self.n_iter == 0:
            self.iter_counter = 1
            self.plot_convergence('accelerated_subgradient', 'accelerated_subgradient_convergence.png')

        else:
            self.iter_counter += 1

        self.beta = self.beta_sq(self.iter_counter, 0.7)
        self.beta_f = self.beta_sq(self.iter_counter, 0.7)


    def nesterov_incremental_sugradient_step(self):
        # Initialize accumulators for subgradients
        initial_cond_subg_total = np.zeros((self.n_agents, nx, 1))
        dynamic_evol_subg_total = np.zeros((self.n_agents, nx, N-1))
        final_state_subg_total = np.zeros((self.n_agents, nx, 1))

        for i, agent in enumerate(self.agents):
            # Circular behaviour
            if i == 0:
                lambda_neg = self.lmbd_fin[-1, :, :]
                momentum_fin_neg = self.momentum_fin[-1, :, :]
            else:
                lambda_neg = self.lmbd_fin[i-1, :, :]
                momentum_fin_neg = self.momentum_fin[i-1, :, :]
            lambda_pos = self.lmbd_fin[i, :, :]
            momentum_fin_pos = self.momentum_fin[i, :, :]

            x, u = agent.solve_local(self.lmbd_ini[i, :, :], self.lmbd_dyn[i, :, :], lambda_pos, lambda_neg)

            self.x[i] = x
            self.u[i] = u

            initial_cond_subg = x[:, 0:1] - self.x_i0[i]
            dynamic_evol_subg = x[:, 1:] - (self.A_i[i] @ x[:, :-1] + self.B_i[i] @ u)
            final_state_neg_subg = -x[:, N-1:N]
            final_state_pos_subg = x[:, N-1:N]

            # Accumulate subgradients
            initial_cond_subg_total[i] = initial_cond_subg
            dynamic_evol_subg_total[i] = dynamic_evol_subg
            final_state_subg_total[i] = final_state_pos_subg
            # Circular behaviour
            if i == 0:
                final_state_subg_total[-1] = final_state_neg_subg
            else:
                final_state_subg_total[i-1] = final_state_neg_subg

            # Calculate momentum
            self.momentum_ini[i, :, :] = self.beta * self.momentum_ini[i, :, :] + self.eta * initial_cond_subg
            self.momentum_dyn[i, :, :] = self.beta * self.momentum_dyn[i, :, :] + self.eta * dynamic_evol_subg
            momentum_fin_pos = self.beta * momentum_fin_pos + self.eta * final_state_pos_subg
            momentum_fin_neg = self.beta * momentum_fin_neg + self.eta * final_state_neg_subg
            # Calculate dual variable update
            self.lmbd_ini[i, :, :] += self.beta * self.momentum_ini[i, :, :]
            self.lmbd_dyn[i, :, :] += self.beta * self.momentum_dyn[i, :, :]
            lambda_pos += self.beta * momentum_fin_pos
            lambda_neg += self.beta * momentum_fin_neg

            # print(f'Dynamic {dynamic_evol_subg}')
            # print(f'Initial {initial_cond_subg}')
            # print(f'Final {final_state_subg}')


        self.subgradient_history["initial"].append(initial_cond_subg_total.copy())
        self.subgradient_history["dynamic"].append(dynamic_evol_subg_total.copy())
        self.subgradient_history["final"].append(final_state_subg_total.copy())

        if self.iter_counter % 200 == 0:
            self.iter_counter = 1
            self.plot_convergence('accelerated_subgradient', 'accelerated_incremental_subgradient_convergence.png')

        else:
            self.iter_counter += 1


    def consensus_subgradient_step(self, param, phi=1):

        # Initialize accumulators for subgradients
        initial_cond_subg_all_agents = np.zeros((self.n_agents, nx, 1))
        dynamic_evol_subg_all_agents = np.zeros((self.n_agents, nx, N-1))
        final_state_subg_all_agents_local = np.zeros((self.n_agents, self.n_agents, nx, 1))

        # STEP 1: Perform local optimizations to obtain subgradient updates
        for i, agent in enumerate(self.agents):
            # Circular behaviour
            if i == 0:
                # Agent has only local information about the subgradients
                lambda_neg = self.lmbd_fin_local[i, -1, :, :]
            else:
                lambda_neg = self.lmbd_fin_local[i, i-1, :, :]
            lambda_pos = self.lmbd_fin_local[i, i, :, :]
            x, u = agent.solve_local(self.lmbd_ini[i, :, :], self.lmbd_dyn[i, :, :], lambda_pos, lambda_neg)

            self.x[i] = x
            self.u[i] = u

        # STEP 2: Do local dual variable update
        for i, _ in enumerate(self.agents):
            x = self.x[i] 
            u = self.u[i] 

            initial_cond_subg = x[:, 0:1] - self.x_i0[i]
            dynamic_evol_subg = x[:, 1:] - (self.A_i[i] @ x[:, :-1] + self.B_i[i] @ u)
            final_state_neg_subg = -x[:, N-1:N]
            final_state_pos_subg = x[:, N-1:N]

            # Save subgradients
            initial_cond_subg_all_agents[i] = initial_cond_subg
            dynamic_evol_subg_all_agents[i] = dynamic_evol_subg
            final_state_subg_all_agents_local[i, i] += final_state_pos_subg
            if i == 0: final_state_subg_all_agents_local[i, -1] += final_state_neg_subg
            else: final_state_subg_all_agents_local[i, i-1] += final_state_neg_subg


            # Update dual variables
            alpha = self.alpha_sq(self.iter_counter, param)
            self.lmbd_ini[i, :, :] += alpha * initial_cond_subg
            self.lmbd_dyn[i, :, :] += alpha * dynamic_evol_subg
            self.lmbd_fin_local[i, i, :, :] += alpha * final_state_pos_subg
            if i == 0: self.lmbd_fin_local[i, -1, :, :] += alpha * final_state_neg_subg
            else: self.lmbd_fin_local[i, i-1, :, :] += alpha * final_state_neg_subg


        # STEP 3: Perform communication with neighbours/consensus iteration phi times
        for _ in range(phi):
            # Save a copy of lmbd_fin_local before consensus update
            lmbd_fin_local_prev = self.lmbd_fin_local.copy()
            self.lmbd_fin_local = np.zeros((self.n_agents, self.n_agents, nx, 1))
            for i in range(self.n_agents):
                for j, comm_w in enumerate(self.consensus_matrix[i, :]):
                    # New dual variable is the weighted average from neighbouring agents
                    self.lmbd_fin_local[i, :, :, :] += comm_w * lmbd_fin_local_prev[j, :, :, :]

        
        final_state_subg_all_agents = np.zeros((self.n_agents, nx, 1))
        for i in range(self.n_agents):
            final_state_subg_all_agents[i] = final_state_subg_all_agents_local[i, i]

        # Save subgradient history
        self.subgradient_history["initial"].append(initial_cond_subg_all_agents.copy())
        self.subgradient_history["dynamic"].append(dynamic_evol_subg_all_agents.copy())
        self.subgradient_history["final"].append(final_state_subg_all_agents.copy())

        if self.iter_counter % self.n_iter == 0:
            self.iter_counter = 1
            self.plot_convergence('consensus_standard_subgradient', 'consensus_subgradient_history.png')
        else:
            self.iter_counter += 1


    def consensus_admm(self, rho):
        # Initialize accumulators for subgradients
        initial_cond_subg_total = np.zeros((self.n_agents, nx, 1))
        dynamic_evol_subg_total = np.zeros((self.n_agents, nx, N-1))
        final_state_subg_total = np.zeros((self.n_agents, nx, 1))

        # Obtain local solutions
        for i, agent in enumerate(self.agents):

            # Perform local updates
            x, u = agent.solve_admm_local(self.lmbd_adm[i], self.x_f, rho)
            # Save solution
            self.x[i] = x
            self.u[i] = u

        # Update consensus variable
        self.x_f = np.zeros((nx, 1))
        for i, _ in enumerate(self.agents):
            x = self.x[i] 
            self.x_f += 1/self.n_agents * (x[:, N-1:N] + 1/rho * self.lmbd_adm[i, :, :])
        # print(self.x_f)
        # Update dual variable
        for i, _ in enumerate(self.agents):
            x = self.x[i] 
            final_state_subg_total[i] = x[:, N-1:N] - self.x_f
            self.lmbd_adm[i] += rho * (x[:, N-1:N] - self.x_f)
        
        self.subgradient_history["initial"].append(initial_cond_subg_total.copy())
        self.subgradient_history["dynamic"].append(dynamic_evol_subg_total.copy())
        self.subgradient_history["final"].append(final_state_subg_total.copy())

        if self.iter_counter % self.n_iter == 0:
            self.iter_counter = 1
            self.plot_convergence('consensus_admm', 'consensus_admm_subgradient_history.png')
            
        else:
            self.iter_counter += 1


    def plot_convergence(self, directory_name, file_name):
        
        os.makedirs(directory_name, exist_ok=True)
        # Use a green-blue colormap for agents, and different line styles for subgradient types
        agent_palette = sns.color_palette("viridis", self.n_agents)
        line_styles = ["-", "--", "-."]
        subgrad_labels = ["Initial", "Dynamic", "Final"]

        # Make the figure taller for a more elongated vertical aspect ratio
        fig_height = 3.5 * self.n_agents  # Increased from 2.5 to 3.5
        fig, axes = plt.subplots(self.n_agents, 1, figsize=(10, fig_height), sharex=True)
        if self.n_agents == 1:
            axes = [axes]
        for i in range(self.n_agents):
            ax = axes[i]
            # Remove top and right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            initial_norm = [np.linalg.norm(sg[i]) for sg in self.subgradient_history["initial"]]
            dynamic_norm = [np.linalg.norm(sg[i]) for sg in self.subgradient_history["dynamic"]]
            final_norm = [np.linalg.norm(sg[i]) for sg in self.subgradient_history["final"]]
            norms = [initial_norm, dynamic_norm, final_norm]
            for j, norm in enumerate(norms):
                ax.semilogy(
                    norm,
                    label=subgrad_labels[j],
                    color=agent_palette[i],
                    linestyle=line_styles[j],
                    linewidth=2.2,
                    alpha=0.95
                )
            ax.set_ylabel("Subgradient Norm", fontsize=16)
            ax.set_title(f"Agent {i+1}", fontsize=16)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.legend(fontsize=13)
        axes[-1].set_xlabel("Iteration", fontsize=16)
        plt.tight_layout(pad=2.0)
        plt.savefig(f"{directory_name}/{file_name}", bbox_inches="tight", dpi=200)
        plt.close()


    def alpha_sq(self, k, a=0.7):
        return a/np.sqrt(k) 
    
    def beta_sq(self, k, a=0.7):
        a = self.beta**2 * a/np.sqrt(k) * (a/np.sqrt(k-1))**(-1)
        return (-a+np.sqrt(a**2+4))/2
            
cord = Coordinator(4, x_i0, A_i, B_i, N, umax)
# cord.coordinator_opt_step()
