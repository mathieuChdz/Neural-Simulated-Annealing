import math
import random
import torch

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
            progress = k / self.n_steps

            # --- temperature ---
            if self.agent:
                temp = self.agent.act_temperature(progress)
                temp = max(0.01, min(100.0, temp))
            else:
                temp *= alpha

            # --- action ---
            if self.agent:
                state_tensor = self.problem.state_to_tensor(state, temp)
                action, log_prob = self.agent.act(state_tensor)
                next_state = self.problem.apply_action(state, action)
            else:
                action = random.randint(0, self.problem.action_space() - 1)
                next_state = self.problem.apply_action(state, action)

            next_energy = self.problem.energy(next_state)
            delta = next_energy - energy

            # Metropolis criterion
            accepted = False

            if delta < 0 or random.random() < math.exp(-delta / temp):
                accepted = True
                state = next_state
                energy = next_energy

                if energy < best_energy:
                    best_state = state.copy()
                    best_energy = energy

            # reward pour PPO
            if self.agent:
                reward = energy - next_energy
                self.agent.store(state_tensor, action, log_prob, reward, False)
            # update PPO every 200 steps
            if self.agent and (k+1) % 200 == 0:
                self.agent.update()

        if self.agent:
            self.agent.update()
            
        return best_state, best_energy