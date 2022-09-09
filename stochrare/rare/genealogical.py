"""
Genealogical Rare Event Algorithms

Examples are the interactive particle system, the GKTL algorithm or Garnier del Moral
"""
# TODO: add references

import numpy as np

class Base():

    def __init__(self, scorefun, ensemble_size=10, timestep=None, k=1):
        self.scorefun = scorefun
        self.ensemble_size = ensemble_size
        self.timestep = timestep
        self.k = k

        self._ensemble = None
        self._weights = None
        self._norm_factor = None

    def initialize_ensemble(self):
        raise NotImplementedError('This is the base class you fool!')

    def prepare_for_next_step(self, ensemble_ids):
        new_ensemble = []
        for j in ensemble_ids:
            new_ensemble.append(self._ensemble[j].copy())
        self._ensemble = new_ensemble

    def select(self) -> list[int]:
        '''
        Selects ensemble members that survive to the next generation (with repetitions) according to `self._weights`
        Returns a list of size `self.ensemble_size` that contains integer values between `0` and `self.ensemble_size - 1`
        '''
        return np.random.choice(self.ensemble_size,size=self.ensemble_size,p=self._weights)

    def propagate_ensemble(self):
        '''
        Propagates all the ensemble members forward in time for `self.timestep`
        '''
        for e in self._ensemble:
            e.update(self.timestep)

    def compute_weights(self):
        self._norm_factor = None
        self._weights = np.array([e.compute_weight(self.scorefun) for e in self._ensemble])

    def normalize_weights(self):
        if self._norm_factor is not None:
            raise ValueError('You are trying to normalize the weights twice')
        self._weights = np.exp(self.k*self._weights)
        self._norm_factor = np.sum(self._weights)
        self._weights /= self._norm_factor

    def step(self):
        '''
        Performs one step of the algorithm
        '''
        self.propagate_ensemble()
        self.compute_weights()
        self.normalize_weights()
        survivors = self.select()
        self.prepare_for_next_step(survivors)


    def run(self, n_iter: int):
        if self._ensemble is None:
            self.initialize_ensemble()

        for i in range(n_iter):
            self.step()