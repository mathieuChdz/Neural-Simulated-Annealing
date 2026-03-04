

class OptimizationProblem:
    def etat_initial(self):
        raise NotImplementedError
        
    def voisinage(self, state):
        raise NotImplementedError
        
    def get_energy(self, state):
        raise NotImplementedError