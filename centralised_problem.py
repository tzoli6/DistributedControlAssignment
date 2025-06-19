import numpy as np
import cvxpy as cp
import seaborn as sns

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

N=20;
umax=7;

nx = 2
nu = 1

x_i0 = [x01, x02, x03, x04]
A_i = [A1, A2, A3, A4]
B_i = [B1, B2, B3, B4]

cost = 0
constraints = []
u = []
x = []

n_agent = 4

# Experiment what if this variable is left out 
# and picevise contrain the finalstate of the agents together
x_f = cp.Variable((2), "final_sate")

for i in range(n_agent):

    # Create input variable and save
    u.append(cp.Variable((nu, N-1), f"input_{i}"))
    x.append(cp.Variable((nx, N  ), f"state_{i}"))

    # Initial condition
    constraints.append(x[i][:, 0:1] == x_i0[i])

    # Dyamic evolution
    for j in range(0, N-1):
        x_next = A_i[i] @ x[i][:, j] + B_i[i] @ u[i][:, j]
        constraints.append(x[i][:, j+1] == x_next)

    # Add state trajectory magnitude into cost
    cost += cp.sum_squares(x[i])
    # Add input trajectory magnitude into cost
    cost += cp.sum_squares(u[i])

    # Add constraint on terminal state
    constraints.append(x[i][:, -1] == x_f)
    # Add conraints on input magnitude
    constraints.append(u[i] <=  umax)
    constraints.append(u[i] >= -umax)



# Solve problem 
cost = cp.Minimize(cost)
centralised_problem = cp.Problem(cost, constraints)
result = centralised_problem.solve()
print(f"Solver status: {centralised_problem.status}")
print(f"Solver optimal value: {centralised_problem.value}")
print(f"Solver result: {result}")
u_solution = []
x_solution = []
print(f"Final state: {x_f.value}")
for i in range(n_agent):
    u_solution.append(u[i].value)
    x_solution.append(x[i].value)
    #print(f"Input trajectory for Agent {i}: \n {u_solution[i]}")
    print(f"State trajectory for Agent {i}: \n {x_solution[i]}")

import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", font_scale=1.2)
import matplotlib
import os
cmap = matplotlib.colormaps.get_cmap("winter")  # green-blue colormap
line_styles = ['-', '--', '-.', ':']
state_labels = [r"$x_1$", r"$x_2$"]


#### START PLOT CENTRALISED TRAJECTORY SOLUTION SEPARATELY ###
fig, axs = plt.subplots(3, 1, figsize=(12, 14))  # Increased height from 10 to 14
plt.rcParams.update({'font.size': 16})  # Set larger font size globally

for agent in range(n_agent):
    color = plt.cm.viridis((agent + 0.5) / n_agent)
    axs[0].plot(range(N), x_solution[agent][0, :].flatten(),
                linestyle=line_styles[agent], color=color, label=f"Agent {agent+1}", linewidth=2.5)
    axs[1].plot(range(N), x_solution[agent][1, :].flatten(),
                linestyle=line_styles[agent], color=color, label=f"Agent {agent+1}", linewidth=2.5)
    axs[2].plot(range(N-1), u_solution[agent][0, :].flatten(),
                linestyle=line_styles[agent], color=color, label=f"Agent {agent+1}", linewidth=2.5)

axs[0].set_ylabel("State 1 [-]", fontsize=19)
axs[1].set_ylabel("State 2 [-]", fontsize=19)
axs[2].set_ylabel("Input [-]", fontsize=19)
axs[2].set_xlabel("Time step", fontsize=19)
for ax in axs:
    ax.legend(fontsize=15)
    ax.grid(True)

plt.tight_layout()
plt.savefig("centralised_state_input_trajectories.png")
plt.show()
### END PLOT CENTRALISED TRAJECTORY SOLUTION SEPARATELY ###


# Read in u_history.npy file
u_dist = np.load("u_history.npy")
x_dist = np.load("x_history.npy")

line_styles = ['-', '--', '-.', ':']
state_labels = [r"$x_1$", r"$x_2$"]

# Use viridis colormap, but use different ranges for distributed and centralised
viridis = plt.cm.get_cmap("viridis")
color_indices_dist = np.linspace(0.1, 0.45, n_agent)   # Lower half for distributed
color_indices_cent = np.linspace(0.55, 0.9, n_agent)   # Upper half for centralised

# Plot state trajectories: all agents stacked vertically, make plots thicker vertically
fig, axs = plt.subplots(n_agent, 1, figsize=(12, 18))  # Increased height from 14 to 18
for agent in range(n_agent):
    color_dist = viridis(color_indices_dist[agent])
    color_cent = viridis(color_indices_cent[agent])
    handles = []
    labels = []
    # Centralised solutions first
    for state_idx in range(nx):
        h_cent, = axs[agent].plot(
            range(N),
            x_solution[agent][state_idx, :].flatten(),
            label=f"Centralised {state_labels[state_idx]}",
            linestyle=line_styles[state_idx],
            color=color_cent,
            linewidth=2.2,
        )
        handles.append(h_cent)
        labels.append(f"Centralised {state_labels[state_idx]}")
    # Distributed solutions after
    for state_idx in range(nx):
        h_dist, = axs[agent].plot(
            range(N),
            x_dist[agent, state_idx, :].flatten(),
            label=f"Distributed {state_labels[state_idx]}",
            linestyle=line_styles[state_idx],
            color=color_dist,
            linewidth=2.2,
        )
        handles.append(h_dist)
        labels.append(f"Distributed {state_labels[state_idx]}")
    axs[agent].set_xlabel("Time step [-]", fontsize=19)
    axs[agent].set_ylabel("State value [-]", fontsize=19)
    axs[agent].set_title(f"Agent {agent+1}", fontsize=19)
    axs[agent].legend(handles, labels, fontsize=16)
    axs[agent].grid(True)
plt.tight_layout()
plt.savefig("centralised_distributed_state_comp_admm_1.png")
plt.show()

# Plot input trajectories: all agents stacked vertically, same aspect as previous plots
fig, axs = plt.subplots(n_agent, 1, figsize=(12, 14))
for agent in range(n_agent):
    color_dist = viridis(color_indices_dist[agent])
    color_cent = viridis(color_indices_cent[agent])
    handles = []
    labels = []
    # Centralised solution first
    h_cent, = axs[agent].plot(
        range(N-1),
        u_solution[agent][0, :].flatten(),
        label="Centralised $u$",
        linestyle='--',
        linewidth=2.2,
        color=color_cent
    )
    handles.append(h_cent)
    labels.append("Centralised $u$")
    # Distributed solution after
    h_dist, = axs[agent].plot(
        range(N-1),
        u_dist[agent, 0, :].flatten(),
        label="Distributed $u$",
        linestyle='-',
        linewidth=2.2,
        color=color_dist
    )
    handles.append(h_dist)
    labels.append("Distributed $u$")
    axs[agent].set_xlabel("Time step [-]", fontsize=19)
    axs[agent].set_ylabel("Input [-]", fontsize=19)
    axs[agent].set_title(f"Agent {agent+1}", fontsize=19)
    axs[agent].legend(handles, labels, fontsize=16)
    axs[agent].grid(True)
plt.tight_layout()
plt.savefig("centralised_distributed_input_comp_admm_1.png")
plt.show()


### START PLOT DISCRIBUTED TRAJECTORY SOLUTION SEPARATELY ###
fig, axs = plt.subplots(3, 1, figsize=(12, 14))
for agent in range(n_agent):
    color = plt.cm.viridis(agent / n_agent)
    axs[0].plot(range(N), x_dist[agent, 0], line_styles[agent], color=color, label=f"Agent {agent+1}", lw=2.5)
    axs[1].plot(range(N), x_dist[agent, 1], line_styles[agent], color=color, label=f"Agent {agent+1}", lw=2.5)
    axs[2].plot(range(N-1), u_dist[agent, 0], line_styles[agent], color=color, label=f"Agent {agent+1}", lw=2.5)
for i, label in enumerate(["State 1 [-]", "State 2 [-]", "Input [-]"]):
    axs[i].set_ylabel(label, fontsize=16)
axs[2].set_xlabel("Time step", fontsize=16)
for ax in axs:
    ax.legend(fontsize=13)
    ax.grid(True)
plt.tight_layout()
plt.savefig("distributed_state_input_trajectories.png")
plt.show()
### END PLOT DISCRIBUTED TRAJECTORY SOLUTION SEPARATELY ###
