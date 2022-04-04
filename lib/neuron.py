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
    V_init: float = -70.        # initial potential [mV]
    V_th: float = -40.          # spike threshold [mV]
    V_reset: float = -55.       # reset potential [mV]
    E_L: float = -70.           # leak reversal potential [mV]
    tau_m: float = 20.          # membrane time constant [ms]
    g_L: float = 2.            # leak conductance [nS]
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
    b: float = 3.          # spiking adaptation
    tau_w: float = 30.     #ms
    delatT: float = 2.
    v_rh: float = -50.

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
        self.states = np.zeros(int(self.Lt/self.dt))
        self.v, self.dv, self.gE, self.gI, self.w = np.zeros(Lt), np.zeros(Lt), np.zeros(Lt), np.zeros(Lt), np.zeros(Lt)

        # init states
        self.v[0] = self.V_init
        self.tr = 0.
        self.spike = 0
        self.spikes = {'times':[], 'senders':[]}

        # initi synapses
        for itype in ['Spikes', 'Istep']:
            for inp in self.inp[itype]:
                inp['buffer'] = deque(int(inp['syn'].delay/dt+1)*[0], int(inp['syn'].delay/dt)+1)

    def update(self, params):
        for k,v in params.items():
            try:
                setattr(self, k, v)
            except:
                "Invalid parameter for LIF neuron."

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
            if weight > 0:
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
        g_L = self.g_L
        dt = self.dt

        # update dynamic variables
        pre_spike_ex, pre_spike_in, I = self.__load_in__()
        self.spike = 0
        if self.tr > 0:                      # freactory period
            self.v[it] = self.V_reset
            self.tr -= 1
        elif self.v[it] >= self.V_th:          # reset voltage and record spike event
            self.v[it] =  self.V_reset
            self.tr = self.tref/dt
            self.spike = 1
            self.spikes['times'].append(it*dt)
            self.spikes['senders'].append(self.idx)
        self.states[it] = self.spike

        # update the synaptic conductance
        self.gE[it+1] = self.gE[it] - (dt/self.tau_syn_E)*self.gE[it] + self.gE_bar*pre_spike_ex
        self.gI[it+1] = self.gI[it] - (dt/self.tau_syn_I)*self.gI[it] + self.gI_bar*np.absolute(pre_spike_in)
            
        # calculate the increment of the membrane potential 
        dv_reg = -(self.v[it]-self.E_L)
        dv_inj = self.delatT * np.exp((self.v[it]-self.v_rh)/self.delatT)
        # dv_inj = 0.
        dv_spk = -(self.gE[it+1]/g_L)*(self.v[it]-self.VE) - (self.gI[it+1]/g_L)*(self.v[it]-self.VI)
        dv_cur = I/g_L
        self.dv[it+1] = (dv_reg + dv_inj + dv_spk + dv_cur) * (dt/self.tau_m)

        # adaptation
        self.w[it+1] = self.w[it] + (-self.w[it] + self.a*(self.v[it]-self.E_L) + self.b*self.tau_w*self.spike)*(dt/self.tau_w)

        # update membrane potential
        self.v[it+1] = self.v[it] + self.dv[it+1] - self.w[it]/g_L

        return self.v[it+1]
    
