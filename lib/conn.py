import numpy as np                 # import numpy

from dataclasses import dataclass, fields

class StaticCon():
    """Static synapse"""
     #! parameters
    weight: float = 1.
    delay: float = 5.   # ms

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for static synapse!"

@dataclass
class Simulator():
    """A simulator control and manages neurons"""
    #! parameters
    dt: float = 0.1     # ms
    cnt: int = 0        # number of devices

    def __post_init__(self):
        """Initialize device list
        """
        self.devidx = []
        self.devices = []

    def __reg__(self, dev):
        """Register a device in active list

        Args:
            dev (any): device e.g. neuron, Poisson generator
        """
        if dev.idx not in self.devidx:
            self.devidx.append(dev.idx)
            self.devices.append(dev)

    def connect(self, src, tar, pars):
        """Connect src pop to tar pop with given parameters

        Args:
            src (list): source population
            tar (list): target population
            pars (dict): connection parameters
        """
        M, N = len(src), len(tar)
        conmat = pars['weight']
        condelay = pars['delay']

        for i in range(M):
            for j in range(N):
                synspec = {'weight': conmat[i,j], 'delay': condelay[i,j]}
                src[i].connect(tar[j], synspec)

    def run(self, T):
        """Run simulation for T

        Args:
            T (float): time length
        """
        ts = list(range(0, int(T/self.dt)))
        # initialize devices for simulation
        for dev in self.devices:
            dev.__initsim__(len(ts), self.dt)

        # run the simulation
        for it in ts[:-1]:
            for dev in self.devices:
                dev.__step__(it)