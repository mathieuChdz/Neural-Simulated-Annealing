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

    def solve(self):

        current_state = self.problem.etat_initial()
        current_energy = self.problem.energy(current_state)

        best_state = current_state
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

                voisin, action = self.problem.voisinage(current_state)

            voisin_energy = self.problem.energy(voisin)

            delta = voisin_energy - current_energy

            if delta < 0:

                accept = True

            else:

                prob = math.exp(-delta / temp)
                accept = random.random() < prob

            if self.agent is not None:

                reward = math.tanh(current_energy - voisin_energy)
                self.agent.store(state_tensor, action, log_prob, reward, False)

            if accept:

                current_state = voisin
                current_energy = voisin_energy

                if current_energy < best_energy:

                    best_energy = current_energy
                    best_state = current_state

            history.append(current_energy)

            temp *= self.alpha

        if self.agent is not None:

            self.agent.update()

        return best_state, best_energy, history