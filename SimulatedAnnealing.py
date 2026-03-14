import math
import random


class SimulatedAnnealing:

    def __init__(self, problem, initial_temp=100.0, final_temp=0.1, n_steps=1000, agent=None):

        self.problem = problem
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.agent = agent

        self.alpha = (final_temp / initial_temp) ** (1 / n_steps)

    def random_neighbor(self, state):

        state = state.copy()

        move_type = random.random()

        # ---- MOVE ITEM ----
        if move_type < 0.5:

            i = random.randint(0, self.problem.n_items - 1)
            j = random.randint(0, self.problem.n_bins - 1)

            state[i] = j

        # ---- SWAP ITEMS ----
        elif move_type < 0.8:

            i = random.randint(0, self.problem.n_items - 1)
            j = random.randint(0, self.problem.n_items - 1)

            state[i], state[j] = state[j], state[i]

        # ---- TRY MERGE BIN ----
        else:

            bin_to_empty = random.randint(0, self.problem.n_bins - 1)

            for i in range(self.problem.n_items):

                if state[i] == bin_to_empty:

                    state[i] = random.randint(0, self.problem.n_bins - 1)

        return state

    def solve(self):

        current_state = self.problem.etat_initial().copy()
        current_energy = self.problem.energy(current_state)

        best_state = current_state.copy()
        best_energy = current_energy

        temp = self.initial_temp

        history = [current_energy]

        for k in range(self.n_steps):

            if self.agent is not None:

                temp_norm = temp / self.initial_temp
                state_tensor = self.problem.state_to_tensor(current_state, temp_norm)

                action, log_prob = self.agent.act(state_tensor)

                voisin = self.problem.apply_action(current_state, action)

            else:

                voisin = self.random_neighbor(current_state)

            voisin_energy = self.problem.energy(voisin)

            delta = voisin_energy - current_energy

            if delta < 0:

                accept = True

            else:

                prob = math.exp(-delta / temp)
                accept = random.random() < prob

            if self.agent is not None:

                reward = current_energy - voisin_energy
                self.agent.store(state_tensor, action, log_prob, reward, False)

            if accept:

                current_state = voisin
                current_energy = voisin_energy

                if current_energy < best_energy:

                    best_energy = current_energy
                    best_state = current_state.copy()

            history.append(current_energy)

            temp *= self.alpha

        if self.agent is not None:

            self.agent.update()

        return best_state, best_energy, history