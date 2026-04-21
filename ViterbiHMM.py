import math
import numpy as np

states = {"s": 0, "E": 1, "5": 2, "I" : 3, "e": 4}
id2state = {0: "s", 1: "E", 2: "5", 3: "I", 4: "e"} 

state_transition_prob = np.array([[0.0, 1.0, 0.0, 0.0, 0.0], 
                                  [0.0, 0.9, 0.1, 0.0, 0.0], 
                                  [0.0, 0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 0.9, 0.1],
                                  [0.0, 0.0, 0.0, 0.0, 0.0]])

emission_nuc_codes = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
emission_probs = np.array([[0.00, 0.00, 0.00, 0.00], 
                           [0.25, 0.25, 0.25, 0.25],
                           [0.05, 0.00, 0.95, 0.00],
                           [0.40, 0.10, 0.10, 0.40],
                           [0.00, 0.00, 0.00, 0.00]])

query_sequence = "CTTCATGTGAAAGCAGACGTAAGTCA"


num_states = len(states)
seq_len = len(query_sequence)

viterbi_value_matrix = np.full((num_states, seq_len), -1000.0)
viterbi_trace_matrix = np.zeros((num_states, seq_len), dtype=int)

for s_idx in range(num_states):
    t_p = state_transition_prob[states["s"]][s_idx]
    e_p = emission_probs[s_idx][emission_nuc_codes[query_sequence[0]]]
    if t_p > 0 and e_p > 0:
        viterbi_value_matrix[s_idx, 0] = math.log(t_p) + math.log(e_p)


def calculate_prob_for_a_node(curr_s, t):
    max_val = -1000.0
    best_prev = 0
    for prev_s in range(num_states):
        t_p = state_transition_prob[prev_s][curr_s]
        e_p = emission_probs[curr_s][emission_nuc_codes[query_sequence[t]]]
        
        if t_p > 0 and e_p > 0:
            
            score = viterbi_value_matrix[prev_s, t-1] + math.log(t_p) + math.log(e_p)
            if score > max_val:
                max_val = score
                best_prev = prev_s
    return max_val, best_prev

for t in range(1, seq_len):
    for s in range(num_states):
        val, trace = calculate_prob_for_a_node(s, t)
        viterbi_value_matrix[s, t] = val
        viterbi_trace_matrix[s, t] = trace

intron_idx = states["I"]
current_state_idx = intron_idx
path = [id2state[current_state_idx]]

for t in range(seq_len - 1, 0, -1):
    current_state_idx = viterbi_trace_matrix[current_state_idx, t]
    path.append(id2state[current_state_idx])

print("".join(reversed(path)))
