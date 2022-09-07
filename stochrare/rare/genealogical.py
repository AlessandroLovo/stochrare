"""
Genealogical Rare Event Algorithms

Examples are the interactive particle system, the GKTL algorithm or Garnier del Moral
"""
# TODO: add references

class Base():

    def __init__(self, scorefun, ensemble_size=10):
        self.scorefun = scorefun
        self.ensemble_size = ensemble_size
        self._ensemble = None

    def initialize_ensemble(self):
        pass

    def select(self):
        pass

    def propagate(self):
        pass

    def run(self, n_iter):
        if self._ensemble is None:
            self.initialize_ensemble()
        for i in range(n_iter):
            self.select()
            self.propagate()