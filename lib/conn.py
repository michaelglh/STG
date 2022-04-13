from dataclasses import dataclass

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

    def __update__(self, spike):
        return self.weight

class GapCon():
    """Static synapse"""
    #! parameters
    weight: float = 1.

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for static synapse!"

    def __update__(self, spike):
        return abs(self.weight)

class FaciCon():
    """Facilitating synapse"""
    #! parameters
    weight: float = 1.
    delay: float = 5.   # ms

    p_init: float = 0.5
    fF: float = 0.01    # facilitation strength
    tau_FP: float = 2e2 # facilitation time constant

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for facilicating synapse!"
        self.prel = self.p_init

    def __update__(self, spike):
        """update synaptic weight when a spike comes

        Args:
            spike (int): spiking or not

        Returns:
            float: synaptic weight
        """      
        self.prel += (self.p_init - self.prel)/self.tau_FP + self.fF*(1-self.p_init)*spike
        
        return self.weight*self.prel/self.p_init

class DeprCon():
    """Depressing synapse"""
    #! parameters
    weight: float = 1.
    delay: float = 5.   # ms

    p_init: float = 0.5
    fD: float = 0.01    # depression scale
    tau_DP: float = 5e2 # depression time constant

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for depressing synapse!"
        self.prel = self.p_init

    def __update__(self, spike):
        """update synaptic weight when a spike comes

        Args:
            spike (int): spiking or not

        Returns:
            float: synaptic weight
        """        
        self.prel += (self.p_init - self.prel)/self.tau_DP - self.fD*self.p_init*spike

        return self.weight*self.prel/self.p_init


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

    def connect(self, src, tar, synspecs):
        """Connect src pop to tar pop with given parameters

        Args:
            src (list): source population
            tar (list): target population
            synspecs (dict): connection parameters
        """
        M, N = len(src), len(tar)

        for i in range(M):
            for j in range(N):
                src[i].connect(tar[j], synspecs[i][j])

    def run(self, T):
        """Run simulation for T

        Args:
            T (float): time length
        """
        ts = list(range(0, int(T/self.dt)))
        # initialize devices for simulation
        for dev in self.devices:
            dev.__initsim__(len(ts), self.dt)

        # run the simulation step by step
        for it in ts[:-1]:
            for dev in self.devices:
                dev.__step__(it)