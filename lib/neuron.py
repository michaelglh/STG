import numpy as np                 # import numpy

import matplotlib.pyplot as plt    # import matplotlib

from dataclasses import dataclass, fields

from .conn import StaticCon

@dataclass
class LIF:
    """Integrate and firing neuron"""
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
        self.inp = {'Poisson':[], 'Istep':[]}

    def connect_syn(self, device, synspec):
        self.inp[device.name].append({'device':device, 'syn':StaticCon(synspec)})

    def connect(self, name, device, weight):
        self.inp[name].append({'device':device, 'weight':weight})

    def initsim(self, Lt, dt):
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
        self.current = 0.

    def load_in(self):
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

    def step(self, it):
        # retrieve parameters
        V_th, V_reset, E_L = self.V_th, self.V_reset, self.E_L
        tau_m, g_L = self.tau_m, self.g_L
        gE_bar, gI_bar = self.gE_bar, self.gI_bar
        VE, VI = self.VE, self.VI
        tau_syn_E, tau_syn_I = self.tau_syn_E, self.tau_syn_I
        tref = self.tref
        dt = self.dt

        # update dynamic variables
        pre_spike_ex, pre_spike_in, I = self.load_in()
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

    def run(self, T, dt):
        '''
        conductance-based LIF dynamics
        
        Expects:
        pars               : parameter dictionary
        I_inj              : injected current [pA]. The injected current here can be a value or an array
        pre_spike_train_ex : spike train input from presynaptic excitatory neuron
        pre_spike_train_in : spike train input from presynaptic inhibitory neuron
        
        Returns:
        rec_spikes : spike times
        rec_v      : mebrane potential
        gE         : postsynaptic excitatory conductance
        gI         : postsynaptic inhibitory conductance
        '''
        
        # # Retrieve parameters
        # V_th, V_reset = self.pars['V_th'], self.pars['V_reset']
        # tau_m, g_L = self.pars['tau_m'], self.pars['g_L']
        # V_init, E_L = self.pars['V_init'], self.pars['E_L']
        # gE_bar, gI_bar = self.pars['gE_bar'], self.pars['gI_bar']
        # VE, VI = self.pars['VE'], self.pars['VI']
        # tau_syn_E, tau_syn_I = self.pars['tau_syn_E'], self.pars['tau_syn_I']
        # tref = self.pars['tref'] 

        V_th, V_reset = self.V_th, self.V_reset
        tau_m, g_L = self.tau_m, self.g_L
        V_init, E_L = self.V_init, self.E_L
        gE_bar, gI_bar = self.gE_bar, self.gI_bar
        VE, VI = self.VE, self.VI
        tau_syn_E, tau_syn_I = self.tau_syn_E, self.tau_syn_I
        tref = self.tref

        range_t = np.arange(0., T, dt)
        Lt = range_t.size

        pre_spike_train_ex = np.zeros((1,Lt))
        pre_spike_train_in = np.zeros((1,Lt))
        I = np.zeros((1,Lt))
        for name, inps in self.inp.items():
            if name == 'Poisson':
                for inp in inps:
                    spike_train = inp['device'].gen()

                    spike_cut = np.zeros((spike_train.shape[0], Lt))
                    if spike_train.shape[1] < Lt:                           # cut to time length Lt
                        spike_cut[:, spike_train.shape[1]] = spike_train
                    else:
                        spike_cut = spike_train[:, :Lt]
                    spike_cut *= np.absolute(inp['weight'])

                    if inp['weight'] > 0:                                   # add to input spike trains
                        pre_spike_train_ex = np.concatenate([pre_spike_train_ex, spike_cut], axis=0)
                    else:
                        pre_spike_train_in = np.concatenate([pre_spike_train_in, spike_cut], axis=0)
            
            if name == 'Istep':
                for inp in inps:
                    I += inp['device'].gen() * inp['weight']
                        
        pre_spike_train_ex_total = pre_spike_train_ex.sum(axis=0)
        pre_spike_train_in_total = pre_spike_train_in.sum(axis=0)
        I = I.sum(axis=0)
        
        # Initialize
        tr = 0.
        v = np.zeros(Lt)
        v[0] = V_init
        gE = np.zeros(Lt)
        gI = np.zeros(Lt)

        # simulation
        rec_spikes = [] # recording spike times
        for it in range(Lt-1):
            if tr >0:
                v[it] = V_reset
                tr = tr-1
            elif v[it] >= V_th:         #reset voltage and record spike event
                rec_spikes.append(it)
                v[it] = V_reset
                tr = tref/dt
            #update the synaptic conductance
            gE[it+1] = gE[it] - (dt/tau_syn_E)*gE[it] + gE_bar*pre_spike_train_ex_total[it+1]
            gI[it+1] = gI[it] - (dt/tau_syn_I)*gI[it] + gI_bar*pre_spike_train_in_total[it+1]
                
            #calculate the increment of the membrane potential
            dv = (-(v[it]-E_L) - (gE[it+1]/g_L)*(v[it]-VE) - (gI[it+1]/g_L)*(v[it]-VI) + I[it]/g_L) * (dt/tau_m)

            #update membrane potential
            v[it+1] = v[it] + dv
            
        rec_spikes = np.array(rec_spikes) * dt
            
        return v, rec_spikes, gE, gI
    
