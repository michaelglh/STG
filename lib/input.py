import numpy as np                 # import numpy

from dataclasses import dataclass, fields

@dataclass
class Poisson_generator():
    """Poisson generator"""
    name: str = 'Poisson'   # type

    #! parameters
    rate: float = 1.        # Hz
    seed: int = None

    #! state of generator
    spike: int = 0          # spiking or not

    # def __post_init__(self):
    #     # Loop through the fields
    #     for field in fields(self):
    #         # If there is a default and the value of the field is none we can assign a value
    #         if not isinstance(field.default, dataclass._MISSING_TYPE) and getattr(self, field.name) is None:
    #             setattr(self, field.name, field.default)

    def initsim(self, Lt, dt):
        self.Lt = Lt
        self.dt = dt

        # set random seed
        if self.seed is None:
            np.random.seed()
        else:
            np.random.seed(seed=self.seed) 

        # generate uniformly distributed random variables
        u_rand = np.random.rand(Lt)
        
        # generate Poisson train
        self.spike_train = 1. * (u_rand < self.rate*self.dt/1e3)

        # initial state
        self.spike = self.spike_train[0]
        
    def step(self, it):
        self.spike = self.spike_train[it]

    def set_pars(self, pars):
        self.pars = pars

    def gen(self, myseed=False):
    
        '''
        Generates poisson trains
        Expects:
        pars       : parameter dictionary
        rate       : noise amplitute [Hz]
        n          : number of Poisson trains
        myseed     : random seed. int or boolean
        
        Returns:
        poisson_train : spike train matrix, ith row represents whether
                        there is a spike in ith spike train over time
                        (1 if spike, 0 otherwise)
        '''
        
        # Retrieve simulation parameters
        dt, range_t = self.pars['dt'], self.pars['range_t']
        Lt = range_t.size

        rate = self.pars['rate']
        n = self.pars['n']
        
        # set random seed
        if myseed:
            np.random.seed(seed=myseed) 
        else:
            np.random.seed()
        
        # generate uniformly distributed random variables
        u_rand = np.random.rand(n, Lt)
        
        # generate Poisson train
        poisson_train = 1. * (u_rand<rate*dt/1000.)
        
        return poisson_train