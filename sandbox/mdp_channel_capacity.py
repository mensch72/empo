import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Define MDP structure
states = ['s0','s1','s2','s3','s4']
actions = ['a1','a2']
terminals = ['s3','s4']
successors = {
    's0': ['s1','s2'],
    's1': ['s3','s4'],
    's2': ['s3','s4'],
    's3': [],
    's4': []
}

# Monte Carlo runs
num_runs = 1000
results = []

nonterm_states = ['s0','s1','s2']

for run in range(num_runs):
    # Sample transition probabilities
    P = {}
    for s in nonterm_states:
        P[s] = {}
        for a in actions:
            probs = np.random.rand(len(successors[s]))
            probs /= probs.sum()
            P[s][a] = dict(zip(successors[s], probs))

    # --- C: recursively computed channel capacity V(s0) ---
    # V(s) = max_{π(s)} ( I(A;S'|s) + E[V(S')|s,π(s)] )
    V = {}
    # Terminal states have value 0
    for s in terminals:
        V[s] = 0

    # Process states in reverse topological order (s1, s2 before s0)
    for s in reversed(nonterm_states):
        best = 0
        s_nexts = successors[s]
        for p_a1 in np.linspace(0, 1, 50):
            p_a2 = 1 - p_a1
            
            # Compute S' distribution given action distribution
            dist_sprime = np.zeros(len(s_nexts))
            for i, s_next in enumerate(s_nexts):
                dist_sprime[i] = p_a1 * P[s]['a1'][s_next] + p_a2 * P[s]['a2'][s_next]
            
            # I(A; S') at this state
            H_sprime = -np.sum(dist_sprime * np.log2(dist_sprime + 1e-12))
            H_sprime_given_a = 0
            for a, p_a in zip(actions, [p_a1, p_a2]):
                probs = np.array([P[s][a][s_next] for s_next in s_nexts])
                H_sprime_given_a += p_a * (-np.sum(probs * np.log2(probs + 1e-12)))
            I_local = H_sprime - H_sprime_given_a
            
            # E[V(S')] - expected value of successor states
            E_V_sprime = sum(dist_sprime[i] * V[s_next] for i, s_next in enumerate(s_nexts))
            
            best = max(best, I_local + E_V_sprime)
        V[s] = best

    C = V['s0']

    # --- log E ---
    q = {}
    for s in nonterm_states:
        q[s] = {}
        for s_next in successors[s]:
            q[s][s_next] = max(P[s]['a1'][s_next], P[s]['a2'][s_next])

    E = {}
    E['s3'] = 1
    E['s4'] = 1
    E['s1'] = q['s1']['s3']*E['s3'] + q['s1']['s4']*E['s4']
    E['s2'] = q['s2']['s3']*E['s3'] + q['s2']['s4']*E['s4']
    E['s0'] = q['s0']['s1']*E['s1'] + q['s0']['s2']*E['s2']

    results.append({'C':C, 'logE':np.log(E['s0'])})

# Scatterplot matrix
df = pd.DataFrame(results)
g = sns.pairplot(df, plot_kws={'alpha': 0.5})
for i, row in enumerate(g.axes):
    for j, ax in enumerate(row):
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
        if i != j:  # Add diagonal line only to scatter plots (not histograms)
            ax.plot([0, 1.2], [0, 1.2], 'k--', alpha=0.5)
plt.show()
