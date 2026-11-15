import numpy as np

states = ['Rainy', 'Sunny']
observations = ['Walk', 'Shop', 'Clean']

start_prob = np.array([0.6, 0.4])

transition = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emission = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

sequence = ['Walk', 'Shop', 'Clean']
obs = [observations.index(x) for x in sequence]

def viterbi(obs):
    T = len(obs)
    N = len(states)
    dp = np.zeros((T, N))
    path = np.zeros((T, N), dtype=int)

    # Initialization
    dp[0] = start_prob * emission[:, obs[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            probabilities = dp[t-1] * transition[:, j]
            path[t, j] = np.argmax(probabilities)
            dp[t, j] = (
                np.max(probabilities)
                * emission[j, obs[t]]
            )

    # Termination
    last_state = np.argmax(dp[-1])
    best_path = [last_state]

    # Backtracking
    for t in range(T-1, 0, -1):
        last_state = path[t, last_state]
        best_path.append(last_state)

    best_path.reverse()
    return [states[i] for i in best_path]

result = viterbi(obs)

print("Observations:", sequence)
print("Most likely hidden states:", result)
