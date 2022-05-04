import numpy as np

from dataclasses import dataclass

@dataclass
class Poisson_generator():
    """Poisson generator"""
    sim: any
    otype: str = 'Spikes'   # type

    #! parameters
    rate: float = 1.        # Hz
    seed: int = None
    start: int = 0
    end: int = int(1e4)

    #! state of generator
    spike: int = 0          # spiking or not

    def __post_init__(self):
        """Initialize device
        """        
        self.idx = self.sim.cnt
        self.sim.cnt += 1
        self.sim.devices.append(self)

    def __initsim__(self, Lt, dt):
        """Initialization for simualtion

        Generate sequence of spike trains following Poisson distribution

        Args:
            Lt (int): number of timesteps
            dt (float): timestep size
        """
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
        self.spike_train[0:self.start] = 0
        self.spike_train[self.end:] = 0

        # initial state
        self.spike = self.spike_train[0]
        
    def __step__(self, it):
        """Update current output state

        Args:
            it (int): current iteration index
        """        
        self.spike = self.spike_train[it]

    def set_pars(self, pars):
        """Update parameters of generator

        Args:
            pars (dict): rate, random seed, etc.
        """        
        self.pars = pars

@dataclass
class Current_injector():
    """Step-wise current injection"""
    sim: any
    otype: str = 'Istep'  # type

    #! parameters
    rate: float = 0.
    start: int = 0
    end: int = int(1e4)

    #! state of generator
    current: int = 0          # spiking or not
    

    def __post_init__(self):
        """Initialize device
        """        
        self.idx = self.sim.cnt
        self.sim.cnt += 1
        self.sim.__reg__(self)

    def __initsim__(self, Lt, dt):
        """Initialization for simualtion

        Generate sequence of current injection following Gaussian distribution

        Args:
            Lt (int): number of timesteps
            dt (float): timestep size
        """
        self.Lt = Lt
        self.dt = dt

        # generate uniformly distributed random variables
        self.Is = np.ones(Lt)*self.rate
        self.Is[0:self.start] = 0
        self.Is[self.end:] = 0

    def __step__(self, it):
        """Update current output state

        Args:
            it (int): current iteration index
        """        
        self.current = self.Is[it]

    def set_pars(self, pars):
        """Update parameters of generator

        Args:
            pars (dict): rate, random seed, etc.
        """        
        self.pars = pars

@dataclass
class Gaussian_generator():
    """Gaussian noise generator"""
    sim: any
    otype: str = 'Istep'  # type

    #! parameters
    mean: float = 0.        # 
    std: float = 1.         # 
    seed: int = None
    start: int = 0
    end: int = int(1e4)

    #! state of generator
    current: int = 0          # spiking or not

    def __post_init__(self):
        """Initialize device
        """        
        self.idx = self.sim.cnt
        self.sim.cnt += 1
        self.sim.__reg__(self)

    def __initsim__(self, Lt, dt):
        """Initialization for simualtion

        Generate sequence of current injection following Gaussian distribution

        Args:
            Lt (int): number of timesteps
            dt (float): timestep size
        """
        self.Lt = Lt
        self.dt = dt

        # set random seed
        if self.seed is None:
            np.random.seed()
        else:
            np.random.seed(seed=self.seed) 

        # generate uniformly distributed random variables
        self.Is = np.random.normal(self.mean, self.std, Lt)
        self.Is[0:self.start] = 0
        self.Is[self.end:] = 0
        
    def __step__(self, it):
        """Update current output state

        Args:
            it (int): current iteration index
        """        
        self.current = self.Is[it]

    def set_pars(self, pars):
        """Update parameters of generator

        Args:
            pars (dict): rate, random seed, etc.
        """        
        self.pars = pars