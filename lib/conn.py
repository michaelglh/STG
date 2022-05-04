from dataclasses import dataclass

class StaticCon():
    """Static synapse"""
    #! parameters
    weight: float = 1.
    delay: float = 5.   # ms
    stype: str = 'chem'

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for static synapse!"
        self.weights = [self.weight]

    def __update__(self, spike):
        self.weights.append(self.weight)

        return self.weight

class GapCon():
    """Static synapse"""
    #! parameters
    weight: float = 1.
    stype: str = 'gap'

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for static synapse!"

    def __update__(self):
        return abs(self.weight)

class FaciCon():
    """Facilitating synapse"""
    #! parameters
    weight: float = 1.
    delay: float = 5.   # ms
    stype: str = 'chem'

    p_init: float = 0.5
    fF: float = 0.5    # facilitation strength
    tau_FP: float = 1e3 # facilitation time constant

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
    stype: str = 'chem'

    p_init: float = 0.5
    fD: float = 0.2     # depression scale
    tau_DP: float = 1e3 # depression time constant

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

class HebbCon():
    """Hebbian synapse"""
    #! parameters
    weight: float = 1.0
    delay: float = 5.   # ms
    stype: str = 'plastic'

    gamma_2: float = 1e-5
    wmax: float = 2.0
    beta: float = 1.0
    gamma_0: float = 1e-4

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for depressing synapse!"
        self.weights = [self.weight]

    def __update__(self, pre, post):
        """update synaptic weight according to instaneous rate of pre and post

        Args:
            pre (neuron): presynaptic neuron
            post (neuron): postsynaptic neuron

        Returns:
            float: synaptic weight
        """

        self.weight +=  (self.gamma_2 * pow(self.wmax - self.weight, self.beta) * pre.rt * post.rt - self.gamma_0 * self.weight)*pre.dt
        self.weight = min(self.weight, self.wmax)

        self.weights.append(self.weight)     

        return self.weight

class CompCon():
    """Competitive hebbian synapse"""
    #! parameters
    weight: float = 1.0
    delay: float = 5.   # ms
    stype: str = 'plastic'

    lam: float = 1e-1

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for depressing synapse!"
        self.weights = [self.weight]

    def __update__(self, pre, post):
        """update synaptic weight according to instaneous rate of pre and post

        Args:
            pre (neuron): presynaptic neuron
            post (neuron): postsynaptic neuron

        Returns:
            float: synaptic weight
        """

        self.weight +=  self.lam*(post.rt*(pre.rt-self.weight))*pre.dt

        self.weights.append(self.weight)     

        return self.weight

class STDPCon():
    """Spike-timing-dependant plastic synapse"""
    #! parameters
    weight: float = 1.0
    delay: float = 5.   # ms
    stype: str = 'plastic'

    trace_x: float = 0.
    tau_x: float = 1e1
    trace_y: float = 0.
    tau_y: float = 1e1

    lam: float = 1e2
    alph: float = 0.5

    def __init__(self, synspec):
        for k,v in synspec.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for depressing synapse!"
        self.weights = [self.weight]
        self.xs = [self.trace_x]
        self.ys = [self.trace_y]

    def __update__(self, pre, post):
        """update synaptic weight according to instaneous rate of pre and post

        Args:
            pre (neuron): presynaptic neuron
            post (neuron): postsynaptic neuron

        Returns:
            float: synaptic weight
        """

        self.trace_x += (-self.trace_x/self.tau_x + pre.spike)*pre.dt
        self.trace_y += (-self.trace_y/self.tau_y + post.spike)*post.dt
        self.weight +=  (-self.lam*self.alph*self.weight*self.trace_y*pre.spike + self.lam*self.trace_x*post.spike)*pre.dt

        self.weights.append(self.weight)
        self.xs.append(self.trace_x)
        self.ys.append(self.trace_y)

        return self.weight


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
        cons = [[None for _ in range(N)] for _ in range(M)]

        for i in range(N):
            for j in range(M):
                cons[i][j] = tar[i].connect(src[j], synspecs[i][j])
        
        return cons

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