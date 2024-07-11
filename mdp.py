class MDP:
    def __init__(self, grid_size, walls, terminals, reward, transition_probs, gamma, epsilon):
        self.grid_size = grid_size
        self.walls = set(walls)
        self.terminals = terminals
        self.reward = reward
        self.transition_probs = transition_probs  # Main, Left, Right, Stay
        self.gamma = gamma
        self.epsilon = epsilon
        self.states = self.init_states()
        

    def init_states(self):
        states = {}
        for x in range(1, self.grid_size[0] + 1):
            for y in range(1, self.grid_size[1] + 1):
                if (x, y) not in self.walls:
                    states[(x, y)] = self.reward
        for terminal in self.terminals:
            states[terminal] = self.terminals[terminal]
        return states

    def actions(self, state):
        if state in [t[:2] for t in self.terminals]:
            return [None]  # No actions possible in terminal states
        return ['N', 'S', 'E', 'W']

    def P(self, state, action):
        if action is None:
            return [(0.0, state)]
        result = []
        x, y = state
        directions = {
            'N': (x, y+1),
            'S': (x, y-1),
            'E': (x+1, y),
            'W': (x-1, y)
        }
        prob_main, prob_left, prob_right, prob_stay = self.transition_probs

        # Main action
        main = directions[action]
        if main in self.states:
            result.append((prob_main, main))

        else:
            result.append((prob_main, state))

        # Left and right actions
        left_right = {
            'N': ['W', 'E'],
            'S': ['E', 'W'],
            'E': ['N', 'S'],
            'W': ['S', 'N']
        }
        for i, direction in enumerate(left_right[action]):
            neighbor = directions[direction]
            if neighbor in self.states:
                result.append((prob_left if i == 0 else prob_right, neighbor))
            else:
                result.append((prob_left if i == 0 else prob_right, state))

        # Stay in place
        result.append((prob_stay, state))
        return result

    def R(self, state):
        return self.states.get(state, self.reward)

def value_iteration(mdp):
    # Initialize utility values for all non-terminal states to zero
    U1 = {s: 0 if s not in mdp.terminals else mdp.terminals[s] for s in mdp.states}
    iter = 1 
    while True:
        U = U1.copy()
        delta = 0
        print("iteration ", iter)
        iter = iter +1   
        print_results(U, mdp)
        # Iterate over each state to update its utility based on potential actions
        for s in mdp.states:
            if s in mdp.terminals:
                continue  # Skip terminal states since their utility does not change
            
            max_utility = float('-inf')  # Start with a very low number
            for a in mdp.actions(s):
                q_value = 0
                for p, s_prime in mdp.P(s,a):
                    reward = mdp.R(s)
                    q_value += p* (reward + mdp.gamma * U[s_prime])
                #total = sum(p * U[s_prime] for p, s_prime in mdp.P(s, a))
                #q_value = mdp.R(s) + mdp.gamma * total  # Q-value calculation
               
                if q_value > max_utility:
                    max_utility = q_value  # Find the maximum Q-value across all actions
            
            U1[s] = max_utility  # Update the utility for state s
            # Check if the change in utility for state s is the largest seen so far
            delta = max(delta, abs(U1[s] - U[s]))
        
        # Convergence criterion check
        if delta <= mdp.epsilon * (1 - mdp.gamma) / mdp.gamma:
            break

    return U

def print_results(U, mdp):
    for y in range(mdp.grid_size[1], 0 , -1):
        for x in range(1, mdp.grid_size[0] + 1):
            if (x,y) in walls:
                print("--------------", end=' ')
            else :
                print(f"{U.get((x, y), '----'): .2f}", end=' ')
        print()


def extract_policy(mdp, U):
    policy = {}
    for s in mdp.states:
        if s in mdp.terminals:
            policy[s] = 'T'  # Mark terminal states with 'T'
        elif s in mdp.walls:
            policy[s] = '-'  # Mark walls with '-'
        else:
            max_utility = float('-inf')
            best_action = None
            for a in mdp.actions(s):
                total = sum(p * U[s_prime] for p, s_prime in mdp.P(s, a))
                q_value = mdp.R(s) + mdp.gamma * total
                if q_value > max_utility:
                    max_utility = q_value
                    best_action = a
            policy[s] = best_action
    return policy

def print_policy(policy, mdp):
    print("Final Policy")
    for y in range(1, mdp.grid_size[1] + 1):
        for x in range(1, mdp.grid_size[0] + 1):
            state = (x, y)
            if state in mdp.walls:
                print(' - ', end='')
            elif state in policy:
                print(f' {policy[state]} ', end='')
            else:
                print('   ', end='')
        print()
    print("\n################ POLICY ITERATION ###########################\n")
    # Re-printing policy to match given format example, usually would not duplicate
    for y in range(1, mdp.grid_size[1] + 1):
        for x in range(1, mdp.grid_size[0] + 1):
            state = (x, y)
            if state in mdp.walls:
                print(' - ', end='')
            elif state in policy:
                print(f' {policy[state]} ', end='')
            else:
                print('   ', end='')
        print()




# Example usage
grid_size = (5, 4)
walls = [(2, 2), (2, 3)]
#terminals = [(4, 2, -1), (4, 3, 1)]
minusTerminal = (5,3)
plusTerminal = (5,4)
plusTerminal2 = (4, 2)
terminals = { minusTerminal : -3, plusTerminal : 2, plusTerminal2 : 1 }

reward = -0.04
transition_probs = (0.8, 0.1, 0.1, 0)
gamma = 0.85
epsilon = 0.001

mdp = MDP(grid_size, walls, terminals, reward, transition_probs, gamma, epsilon)
u = value_iteration(mdp)
#print_results(u, mdp)
policy = extract_policy(mdp,u)
print_policy(policy,mdp)