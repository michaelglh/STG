import numpy as np                 # import numpy

import matplotlib.pyplot as plt    # import matplotlib

from dataclasses import dataclass
from collections import deque

from .conn import StaticCon, FaciCon, DeprCon

@dataclass
class LIF:
    """Integrate and firing neuron
    Default value to typical neuron parameters and static synapse parameters
    """
    sim: any

    name: str = 'LIF'           # type
    otype: str = 'Spikes'

    #! neuron parameters
    V_init: float = -75.        # initial potential [mV]
    V_th: float = -55.          # spike threshold [mV]
    V_reset: float = -65.       # reset potential [mV]
    E_L: float = -75.           # leak reversal potential [mV]
    tau_m: float = 10.          # membrane time constant [ms]
    g_L: float = 10.            # leak conductance [nS]
    tref: float = 2.            # refractory time (ms)

    #! synaptic parameters
    gE_bar: float = 3.       #nS
    VE: float = 0.           #mV
    tau_syn_E: float = 2.    #ms
    gI_bar: float = 3.       #nS
    VI: float = -80.         #mV
    tau_syn_I: float = 5.    #ms

    #! adaptation
    a: float = 0.           # subthreshold adaptation
    b: float = 25.          # spiking adaptation
    tau_k: float = 100.     #ms

    #! state of neuron
    spike: int = 0          # spiking or not

    def __post_init__(self):
        """Initialize device
        """        
        self.idx = self.sim.cnt
        self.sim.cnt += 1
        self.sim.devices.append(self)

        self.inp = {'Spikes':[], 'Istep':[]}

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
        self.v, self.dv, self.gE, self.gI, self.w = np.zeros(Lt), np.zeros(Lt), np.zeros(Lt), np.zeros(Lt), np.zeros(Lt)

        # init states
        self.v[0] = self.V_init
        self.tr = 0.
        self.spike = 0

        # initi synapses
        for itype in ['Spikes', 'Istep']:
            for inp in self.inp[itype]:
                inp['buffer'] = deque(int(inp['syn'].delay/dt+1)*[0], int(inp['syn'].delay/dt)+1)

    def connect(self, device, synspec):
        """Connect device to neuron with specification on synapse

        Args:
            device (instance): input device
            synspec (dict): weight, delay, etc.
        """        
        if synspec['ctype'] == 'static':
            self.inp[device.otype].append({'device':device, 'syn':StaticCon(synspec)})
        elif synspec['ctype'] == 'facilitate':
            self.inp[device.otype].append({'device':device, 'syn':FaciCon(synspec)})
        elif synspec['ctype'] == 'depress':
            self.inp[device.otype].append({'device':device, 'syn':DeprCon(synspec)})

    def __load_in__(self):
        """Load input from input devices

        Returns:
            excitatory spikes, inhibitory spikes and current flow
        """
        # spike trains
        pre_spike_ex = 0.
        pre_spike_in = 0.
        for inp in self.inp['Spikes']:
            # update input buffer
            inp['buffer'].appendleft(inp['device'].spike)
            spike_in = inp['buffer'][-1]
            # update input weight
            weight = inp['syn'].__update__(spike_in)
            if inp['syn'].weight > 0:
                pre_spike_ex += spike_in * weight
            else:
                pre_spike_in += spike_in * weight

        # current injections
        I = 0.
        for inp in self.inp['Istep']:
            inp['buffer'].appendleft(inp['device'].current)
            I += inp['buffer'][-1] * inp['syn'].weight

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
        self.dv[it+1] = dv

        # adaptation
        self.w[it+1] = self.w[it] + (-self.w[it] + self.a*(self.v[it]-E_L) + self.b*self.tau_k*self.spike)* (dt/self.tau_k)
        self.w[it+1] = 0

        # update membrane potential
        self.v[it+1] = self.v[it] + dv - self.w[it]/g_L

        return dv
    
