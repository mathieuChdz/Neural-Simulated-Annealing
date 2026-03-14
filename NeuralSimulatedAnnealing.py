import math
import random


class NeuralSimulatedAnnealing:

    def __init__(self, problem, n_steps=1000, agent=None):

        self.problem = problem
        self.n_steps = n_steps
        self.agent = agent

    def solve(self):

        state = self.problem.etat_initial()
        energy = self.problem.energy(state)

        best_state = state.copy()
        best_energy = energy

        temp = 100.0
        final_temp = 0.1

        alpha = (final_temp / temp) ** (1 / self.n_steps)

        for k in range(self.n_steps):

            temp_norm = temp / 100.0

            if self.agent:

                state_tensor = self.problem.state_to_tensor(state, temp_norm)

                action, log_prob = self.agent.act(state_tensor)

                next_state = self.problem.apply_action(state, action)

            else:

                i = random.randint(0, self.problem.n_items - 1)
                j = random.randint(0, self.problem.n_bins - 1)

                next_state = self.problem.apply_action(state, (i, j))

            next_energy = self.problem.energy(next_state)

            delta = next_energy - energy

            accept = False

            if delta < 0 or random.random() < math.exp(-delta / temp):

                accept = True

            old_energy = energy

            if accept:

                state = next_state
                energy = next_energy

                if energy < best_energy:
                    best_state = state.copy()
                    best_energy = energy

            reward = old_energy - energy

            if self.agent:
                self.agent.store(
                    state_tensor,
                    action,
                    log_prob,
                    reward,
                    False
                )
                
            temp *= alpha

        if self.agent:

            self.agent.update()

        return best_state, best_energy