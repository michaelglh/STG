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

    def __post_init__(self):
        self.devices = []

    def reg(self, devices):
        """Register devices included in simulation

        Args:
            devices (instances): neurons, input generators, etc
        """        
        self.devices += devices

    def run(self, T):
        ts = list(range(0, int(T/self.dt)))
        # initialize devices for simulation
        for dev in self.devices:
            dev.__initsim__(len(ts), self.dt)

        # run the simulation
        for it in ts[:-1]:
            for dev in self.devices:
                dev.__step__(it)