import numpy as np

# Hidden states: Rainy, Sunny
states = ['Rainy', 'Sunny']

# Observations: Walk, Shop, Clean
observations = ['Walk', 'Shop', 'Clean']

# Initial probabilities
start_prob = np.array([0.6, 0.4])

# Transition probabilities
transition = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emission probabilities
emission = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

# Observation sequence
sequence = ['Walk', 'Shop', 'Clean']

# Convert observations to indexes
obs_index = [observations.index(x) for x in sequence]

# Forward algorithm
def forward(obs):
    n_states = len(states)
    T = len(obs)
    alpha = np.zeros((T, n_states))

    # Initialization
    alpha[0] = start_prob * emission[:, obs[0]]

    # Recursion
    for t in range(1, T):
        for j in range(n_states):
            alpha[t, j] = (
                np.sum(alpha[t-1] * transition[:, j])
                * emission[j, obs[t]]
            )

    # Total probability of observations
    return np.sum(alpha[-1])

probability = forward(obs_index)

print("Observation sequence:", sequence)
print("Probability:", probability)
