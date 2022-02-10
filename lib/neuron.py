import numpy as np                 # import numpy

import matplotlib.pyplot as plt    # import matplotlib

from dataclasses import dataclass, fields

from .conn import StaticCon

@dataclass
class LIF:
    """Integrate and firing neuron
    Default value to typical neuron parameters and static synapse parameters
    """
    name: str = 'LIF'           # type

    #! neuron parameters
    V_init: float = -75.       # initial potential [mV]
    V_th: float = -55.        # spike threshold [mV]
    V_reset: float = -65.      # reset potential [mV]
    E_L: float = -75.          # leak reversal potential [mV]
    tau_m: float = 10.        # membrane time constant [ms]
    g_L: float = 10.          # leak conductance [nS]
    tref: float = 2.         # refractory time (ms)

    #! synaptic parameters
    gE_bar: float = 3.       #nS
    VE: float = 0.           #mV
    tau_syn_E: float = 2.    #ms
    gI_bar: float = 3.       #nS
    VI: float = -80.         #mV
    tau_syn_I: float = 5.    #ms

    #! state of neuron
    spike: int = 0          # spiking or not

    def __post_init__(self):
        """Initialize input device lists
        """        
        self.inp = {'Poisson':[], 'Istep':[]}

    def __initsim__(self, Lt, dt):
        """Initialization for simualtion

        Recording arrays of potential, conductances, spiking state etc

        Init state of the neuron

        Args:
            Lt (int): number of timesteps
            dt (float): timestep size
        """
        # simulation setting
        self.dt = dt
        self.Lt = Lt

        # dynamics variables
        self.rec_spikes = []
        self.v, self.gE, self.gI = np.zeros(Lt), np.zeros(Lt), np.zeros(Lt)

        # init states
        self.v[0] = self.V_init
        self.tr = 0.
        self.spike = 0

    def connect(self, device, synspec):
        """Connect device to neuron with specification on synapse

        Args:
            device (instance): input device
            synspec (dict): weight, delay, etc.
        """        
        self.inp[device.name].append({'device':device, 'syn':StaticCon(synspec)})

    def __load_in__(self):
        """Load input from input devices

        Returns:
            excitatory spikes, inhibitory spikes and current flow
        """        
        # spike trains
        pre_spike_ex = 0.
        pre_spike_in = 0.
        for inp in self.inp['Poisson']:
            spike_in = inp['device'].spike
            if inp['syn'].weight > 0:
                pre_spike_ex += spike_in * inp['syn'].weight
            else:
                pre_spike_in += spike_in * inp['syn'].weight

        # current injections
        I = 0.
        for inp in self.inp['Istep']:
            I += inp['device'].current * inp['syn'].weight

        return pre_spike_ex, pre_spike_in, I

    def __step__(self, it):
        """Simulation for one step

        Args:
            it (int): current iteration index

        Returns:
            float: change in potential
        """        
        # retrieve parameters
        V_th, V_reset, E_L = self.V_th, self.V_reset, self.E_L
        tau_m, g_L = self.tau_m, self.g_L
        gE_bar, gI_bar = self.gE_bar, self.gI_bar
        VE, VI = self.VE, self.VI
        tau_syn_E, tau_syn_I = self.tau_syn_E, self.tau_syn_I
        tref = self.tref
        dt = self.dt

        # update dynamic variables
        pre_spike_ex, pre_spike_in, I = self.__load_in__()
        self.spike = 0
        if self.tr > 0:                      # freactory period
            self.v[it] = V_reset
            self.tr -= 1
        elif self.v[it] >= V_th:          # reset voltage and record spike event
            self.rec_spikes.append(it*dt)
            self.v[it] =  V_reset
            self.tr = tref/dt
            self.spike = 1

        # update the synaptic conductance
        self.gE[it+1] = self.gE[it] - (dt/tau_syn_E)*self.gE[it] + gE_bar*pre_spike_ex
        self.gI[it+1] = self.gI[it] - (dt/tau_syn_I)*self.gI[it] + gI_bar*np.absolute(pre_spike_in)
            
        # calculate the increment of the membrane potential
        dv = (-(self.v[it]-E_L) - (self.gE[it+1]/g_L)*(self.v[it]-VE) - (self.gI[it+1]/g_L)*(self.v[it]-VI) + I/g_L) * (dt/tau_m)

        # update membrane potential
        self.v[it+1] = self.v[it] + dv

        return dv
    
