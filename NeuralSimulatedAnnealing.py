import math
import random
import torch

class NeuralSimulatedAnnealing:
    def __init__(self, problem, initial_temp=100.0, final_temp=0.1, n_steps=1000, agent=None):
        self.problem = problem
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.n_steps = n_steps
        self.alpha = (final_temp / initial_temp) ** (1 / n_steps)
        self.agent = agent

    def solve(self):
        current_state = self.problem.etat_initial()
        current_energy = self.problem.energy(current_state)

        best_state = current_state.copy()
        best_energy = current_energy

        temp = self.initial_temp

        for k in range(self.n_steps):

            # === Construction du state tensor pour item/bin factorized ===
            if self.agent is not None:
                state_tensor = {
                    'weights': self.problem.items,
                    'bins': current_state,
                    'bin_remaining': [self.problem.bin_capacity - sum(
                        self.problem.items[idx] for idx, b in enumerate(current_state) if b == j
                    ) for j in range(self.problem.n_bins)],
                    'temp': temp
                }
                action, log_prob = self.agent.act(state_tensor)
                next_state = self.problem.apply_action(current_state, action)
            else:
                # Vanilla random
                action = random.randint(0, self.problem.action_space() - 1)
                next_state = self.problem.apply_action(current_state, action)

            next_energy = self.problem.energy(next_state)
            delta_e = next_energy - current_energy

            # Metropolis
            if delta_e < 0 or random.random() < math.exp(-delta_e / temp):
                current_state = next_state
                current_energy = next_energy

                if self.agent is not None:
                    reward = math.tanh(-delta_e)
                    self.agent.store(state_tensor, action, log_prob, reward, False)

                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy

            # Temperature update
            temp *= self.alpha

            # Optional: update every 200 steps
            if self.agent is not None and (k+1) % 200 == 0:
                self.agent.update()

        if self.agent is not None:
            self.agent.update()

        return best_state, best_energy