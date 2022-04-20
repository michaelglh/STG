import numpy as np                 # import numpy
import matplotlib.pyplot as plt    # import matplotlib

def plot_volt_trace(pars, v, sp, c='b', axis=None):
    '''
    Plot trajetory of membrane potential for a single neuron
  
    Expects:
    pars   : parameter dictionary
    v      : volt trajetory
    sp     : spike train
  
    Returns:
    figure of the membrane potential trajetory for a single neuron
    '''

    V_th = pars['V_th']
    dt, range_t = pars['dt'], pars['range_t']
    if sp.size:
       sp_num = (sp/dt).astype(int)-1
       v[sp_num] += 10

    if axis is None:
        plt.plot(range_t, v, color=c)
        plt.axhline(V_th, 0, 1, color='k', ls='--')
        plt.xlabel('Time (ms)')
        plt.ylabel('V (mV)')
    else:
        axis.plot(range_t, v, color=c)
        axis.axhline(V_th, 0, 1, color='k', ls='--')
        axis.set_xlabel('Time (ms)')
        axis.set_ylabel('V (mV)')

def pair_volt_trace(pars, vs, sps, axis=None):
    '''
    Plot trajetory of membrane potential for a pair of neurons
  
    Expects:
    pars   : parameter dictionary
    vs     : volt trajetories
    sps    : spike trains
  
    Returns:
    figure of the membrane potential trajetory for a single neuron
    '''

    V_ths = pars['V_ths']
    dt = pars['dt']

    cs = ['r', 'b']

    if axis is None:
        plt.plot(vs[0], vs[1], color='k')
        plt.axhline(V_ths[0], 0, 1, color=cs[0], ls='--')
        plt.axvline(V_ths[1], 0, 1, color=cs[1], ls='--')
        plt.xlabel('V (mV)')
        plt.ylabel('V (mV)')
    else:
        axis.plot(vs[0], vs[1], color='k')
        axis.axhline(V_ths[0], 0, 1, color=cs[0], ls='--')
        axis.axvline(V_ths[1], 0, 1, color=cs[1], ls='--')
        axis.set_xlabel('V (mV)')
        axis.set_ylabel('V (mV)')

def plot_raster_Poisson(range_t, spike_train, n):
    '''
    Generates poisson trains
    Expects:
    range_t     : time sequence 
    spike_train : binary spike trains, with shape (N, Lt)
    n           : number of Poisson trains plot
    
    Returns:
    Raster_plot of the spike train
    '''
    #find the number of all the spike trains
    N = spike_train.shape[0]

    # n should smaller than N:
    if n > N:
       print('The number n exceeds the size of spike trains')
       print('The number n is set to be the size of spike trains')
       n = N

    #plot rater
    i = 0
    while i< n:
        if spike_train[i, :].sum()>0.:
          t_sp = range_t[spike_train[i, :]>0.5] #spike times 
          plt.plot(t_sp, i*np.ones(len(t_sp)), 'k|', ms=10, markeredgewidth=2)
        i += 1
    plt.xlim([range_t[0], range_t[-1]])
    plt.ylim([-0.5, n+0.5])    
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Neuron ID', fontsize=12);