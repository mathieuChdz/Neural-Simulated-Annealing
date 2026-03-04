class NSAProblem:

    def etat_initial(self):
        pass

    def energy(self, state):
        pass

    def apply_action(self, state, action):
        pass

    def state_to_tensor(self, state, temperature):
        pass

    def action_space(self):
        pass