import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import math
from tqdm import tqdm

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ms = 1e-3
mV = 1e-3
MOhm = 1e6
nA = 1e-9
nS = 1e-9

##########################		Part A: Question 1		###########################

def time_steps(max_t, dt):
	return np.arange(0, max_t + dt / 2, dt)

def dV_dt(V):
	E_l = -70 * mV
	R_m = 10 * MOhm
	I_e = 3.1 * nA
	tau_m = 10 * ms

	return ( E_l - V + R_m * I_e ) / tau_m

def Euler_simple(max_t, dt, V_rest, V_th):

	time = time_steps(max_t, dt)

	voltage = np.zeros(len(time))
	voltage[0] = V_rest
	for t in range(len(time) - 1):
	    voltage[t+1] = voltage[t] + dV_dt(voltage[t]) * dt
	    if voltage[t+1] >= V_th:
	    	voltage[t+1] = V_rest
	
	return time, voltage

def plotA1(V_rest, V_th):

	time, voltage = Euler_simple(   max_t = 1,
									dt = 0.25 * ms,
									V_rest = V_rest,
									V_th = V_th)

	f = plt.figure(1)

	plt.plot(time, voltage / mV, color = "black", label = "Voltage")
	plt.axhline(V_th / mV, linestyle = "--", alpha = 0.7, label = "Threshold", color = "red")
	plt.axhline(V_rest / mV, linestyle = "--", alpha = 0.7, label = "Reset voltage", color = 'blue')
	plt.title(r'\textbf{Integrate-and-fire model - Voltage against time}', fontsize=18)
	plt.xlabel(r'Time (s)', fontsize=16)
	plt.ylabel(r'Voltage (mV)', fontsize=16)
	plt.ylim([-75, -20])
	plt.grid(True, linestyle='-.')
	plt.axvline(linewidth=1, color='black')
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.legend(prop={"size":16})
	plt.show()

# plotA1(V_rest = -70 * mV, V_th = -40 * mV)

###########################		Part A: Question 2		###########################

def conductance_dV_dt(V, s, E_s):
	E_l = -70 * mV
	Rm_Ie = 18 * mV
	Rm_gs = 0.15
	tau_m = 20 * ms

	return ( E_l - V + Rm_Ie + Rm_gs * s * (E_s - V) ) / tau_m

def ds_dt(s):
	tau_s = 10 * ms
	return -s / tau_s

def Euler_2_neuron(max_t, dt, V_rest, V_th, E_s, P):
	
	time = time_steps(max_t, dt)

	neuron_V = np.zeros((len(time),2))
	neuron_V[0,:] = [uniform(V_rest, V_th), uniform(V_rest, V_th)]

	s = np.zeros((len(time),2))

	for t in range(len(time)-1):
	    neuron_V[t+1,:] = neuron_V[t,:] + conductance_dV_dt(neuron_V[t,:], s[t,:], E_s) * dt
	    s[t+1,:] = s[t,:] + ds_dt(s[t,:]) * dt

	    if neuron_V[t+1,0] >= V_th:
	    	neuron_V[t+1,0] = V_rest
	    	s[t+1,1] = s[t+1,1] + P

	    if neuron_V[t+1,1] >= V_th:
	    	neuron_V[t+1,1] = V_rest
	    	s[t+1,0] = s[t+1,0] + P

	return time, neuron_V

def plotA2(V_rest, V_th):

	time, neuron_V = Euler_2_neuron(max_t = 1,
									dt = 0.25 * ms,
									V_rest = V_rest,
									V_th = V_th,
									E_s = [0, -80*mV][0], # excitatory and inhibitory synapses
									P = 0.5)


	f = plt.figure(2)

	plt.plot(time, neuron_V[:,0] / mV, color = "firebrick", label = "Neuron 1")
	plt.plot(time, neuron_V[:,1] / mV, color = "royalblue", label = "Neuron 2")
	plt.axhline(V_th / mV, linestyle = "--", alpha = 0.7, label = "Threshold", color = "red")
	plt.axhline(V_rest / mV, linestyle = "--", alpha = 0.7, label = "Reset voltage", color = 'blue')
	plt.title(r'\textbf{Simulating two neurons with synaptic connections}', fontsize=18)
	plt.xlabel(r'Time (s)', fontsize=16)
	plt.ylabel(r'Voltage (mV)', fontsize=16)
	plt.ylim([-85, -30])
	plt.grid(True, linestyle='-.')
	plt.axvline(linewidth=1, color='black')
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.legend(prop={"size":16})
	plt.show()

# plotA2(V_rest = -80 * mV, V_th = -54 * mV)

###########################		Part B: Question 1		###########################

def B_dV_dt(V, g_bar_s):
	E_l = -65 * mV
	R_m = 100 * MOhm
	E_s = 0
	tau_m = 10 * ms

	return ( E_l - V + R_m * g_bar_s * (E_s - V) ) / tau_m

def ds_dt_Nsyn(s):
	tau_s = 2 * ms
	return - s / tau_s

def spikes_binary(max_t, dt, N, avg_fire_rate, correlated, r_0, B, f): # number of synapses
	
	time = time_steps(max_t, dt)
	spikes = np.zeros((len(time), N))

	for i in range(len(time)):
		spikes[i,:] = np.random.uniform(0,1,N)

		for j in range(len(spikes[i,:])):
			if spikes[i,j] < avg_fire_rate * dt:
				spikes[i,j] = 1
			else:
				spikes[i,j] = 0 # n columns of binary spikes over 0 to 1 in dt intervals

		if correlated:
			avg_fire_rate = r_0 + B * math.sin( 2 * math.pi * f * (i * dt) )

	return spikes

def time_recent_spike(spikes, dt):
	
	train = np.zeros(spikes.shape)
	for count, values in enumerate(spikes[1:,:]):

		train[count+1,:] = values * (count+1) * dt

		for j in range(len(train[0,:])):
			if train[count+1,j] == 0:
				train[count+1,j] = train[count,j]

	return train # returns most recent spike for each synapse at each time point

def STDP(t_pre, t_post, g_bar, switch):

	if switch:

		delta_t = t_post - t_pre

		A_plus = 0.2 * nS
		A_minus = 0.25 * nS
		tau_plus = 20 * ms
		tau_minus = 20 * ms

		f_t = np.zeros(len(delta_t))
		for i in range(len(delta_t)):
			if delta_t[i] > 0:
				f_t[i] = A_plus * math.exp( - abs(delta_t[i]) / tau_plus )
			else:
				f_t[i] = - A_minus * math.exp( - abs(delta_t[i]) / tau_minus )

		g_bar  = g_bar + f_t

		for i in range(len(g_bar)):
			if g_bar[i] < 0:
				g_bar[i] = 0
			if g_bar[i] > 4 * nS:
				g_bar[i] = 4 * nS

	return g_bar

def Euler_N_synapses(max_t, dt, V_rest, V_th, P, N, avg_fire_rate, switch, correlated, r_0, B, f):

	time = time_steps(max_t, dt)
	spikes = spikes_binary(max_t, dt, N, avg_fire_rate, correlated, r_0, B, f)
	train = time_recent_spike(spikes, dt)

	neuron_V = np.zeros(len(time))
	neuron_V[0] = V_rest

	g_bar = np.ones(N) * 4 * nS

	s = np.zeros((len(time), N))

	g_bar_s = sum(g_bar * s[0,:])

	count = np.zeros( int(max_t / dt) )
	t_post = 0
	for t in range(len(time)-1):

		neuron_V[t+1] = neuron_V[t] + B_dV_dt(neuron_V[t], g_bar_s) * dt

		if neuron_V[t+1] >= V_th:
			neuron_V[t+1] = V_rest
			count[t+1] = 1
			t_post = (t+1) * dt
			t_pre = train[t+1,:]
			g_bar = STDP(t_pre, t_post, g_bar, switch)

		for j in range(len(spikes[t+1,:])):
			if spikes[t+1,j] == 1:
				t_pre = (t+1) * dt
				g_bar[j] = STDP(np.array([t_pre]), t_post, g_bar[j], switch)

		s[t+1,:] = s[t,:] + ds_dt_Nsyn(s[t,:]) * dt + spikes[t,:] * P

		g_bar_s = sum(g_bar * s[t+1,:])

	return time, neuron_V, g_bar, count

def plotB1(V_rest, V_th, switch):

	time, neuron_V, g_bar, count = Euler_N_synapses(max_t = 1,
													dt = 0.25 * ms,
													V_rest = V_rest,
													V_th = V_th,
													P = 0.5,
													N = 40,
													avg_fire_rate = 15,
													switch = switch,
													correlated = False,
													r_0 = 20,
													B = 0,
													f = 10)

	f = plt.figure(3)

	plt.plot(time, neuron_V / mV, color = "black", label = "Voltage")
	plt.axhline(V_th / mV, linestyle = "--", alpha = 0.7, label = "Threshold", color = "red")
	plt.axhline(V_rest / mV, linestyle = "--", alpha = 0.7, label = "Reset voltage", color = 'blue')
	plt.title(r'\textbf{Integrate-and-fire model - Voltage against time}', fontsize=18)
	plt.xlabel(r'Time (s)', fontsize=16)
	plt.ylabel(r'Voltage (mV)', fontsize=16)
	plt.ylim([-75, -20])
	plt.grid(True, linestyle='-.')
	plt.axvline(linewidth=1, color='black')
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.legend(prop={"size":16})
	plt.show()

###########################		Part B: Question 2		###########################

def synaptic_weights(switch, avg_fire_rate, correlated, B):

	g_bar_vec = []
	for i in tqdm(range(30)):

		time, neuron_V, g_bar, count = Euler_N_synapses(max_t = 300,
														dt = 0.25 * ms,
														V_rest = -65 * mV,
														V_th = -50 * mV,
														P = 0.5,
														N = 40,
														avg_fire_rate = avg_fire_rate,
														switch = switch,
														correlated = correlated,
														r_0 = 20,
														B = B,
														f = 10)
		g_bar_vec += list(g_bar)

	f = plt.figure(4)

	plt.hist(g_bar_vec, bins = 30, density = True)
	plt.title(r'\textbf{Histogram of the steady-state synaptic weights}', fontsize=18)
	plt.xlabel(r'Steady-state synaptic weights (nS)', fontsize=16)
	plt.ylabel(r'Probability density', fontsize=16)
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.show()

def average_firing_rate(switch, max_t, plot):

	AFR = np.zeros((10, 30))
	for i in tqdm(range(10)):
		time, neuron_V, g_bar, count = Euler_N_synapses(max_t = max_t,
														dt = 0.25 * ms,
														V_rest = -65 * mV,
														V_th = -50 * mV,
														P = 0.5,
														N = 40,
														avg_fire_rate = 15,
														switch = switch,
														correlated = False,
														r_0 = 20,
														B = 0,
														f = 10)

		
		count = np.array( np.split(count, int( max_t / 10) ) )
		AFR[i,:] = np.sum(count, axis = 1) / 10

	AFR = np.mean(AFR, axis = 0)

	last_30s = np.mean( AFR[-4:-1] )

	if plot:

		f = plt.figure(5)

		plt.plot(np.linspace(5,295,30), AFR, 'x-', color = "black")
		plt.title(r'\textbf{Average firing rate of the postsynaptic neuron over time bins}', fontsize=18)
		plt.xlabel(r'Bin centres (s)', fontsize=16)
		plt.ylabel(r'Average firing rate (Hz)', fontsize=16)
		plt.xticks(np.linspace(0,300,31))
		plt.grid(True, linestyle='-.')
		plt.tick_params(labelcolor='black', labelsize='13', width=1)
		plt.show()

	return last_30s

###########################		Part B: Question 3		###########################

def vary_input_FR(switch):

	last_30s = np.zeros((5,11))

	for i in tqdm(range(5)):
		for I_FR in np.linspace(10,20,11):
		
			time, neuron_V, g_bar, count = Euler_N_synapses(max_t = 300,
															dt = 0.25 * ms,
															V_rest = -65 * mV,
															V_th = -50 * mV,
															P = 0.5,
															N = 40,
															avg_fire_rate = I_FR,
															switch = switch,
															correlated = False,
															r_0 = 20,
															B = 0,
															f = 10)


			count = np.array( np.split(count, 30) )
			AFR_output = np.sum(count, axis = 1) / 10

			last_30s[ i, int(I_FR - 10) ] = np.mean( AFR_output[-4:-1] )

	last_30s = np.mean(last_30s, axis = 0)

	f = plt.figure(6)

	plt.plot( np.linspace(10,20,11) , last_30s, 'x-', color = "black")
	plt.title(r'\textbf{Mean output firing rate as a function of the input firing rates}', fontsize=18)
	plt.xlabel(r'Input firing rate (Hz)', fontsize=16)
	plt.ylabel(r'Mean output firing rate (Hz)', fontsize=16)
	plt.grid(True, linestyle='-.')
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.show()

###########################		Part B: Question 4		###########################

def mean_var():

	mean_vec = np.zeros((10, 5))
	sd_vec = np.zeros((10, 5))

	for i in tqdm(range(10)):
		for B in np.linspace(0,20,5):

			time, neuron_V, g_bar, count = Euler_N_synapses(max_t = 300,
															dt = 0.25 * ms,
															V_rest = -65 * mV,
															V_th = -50 * mV,
															P = 0.5,
															N = 40,
															avg_fire_rate = 20,
															switch = True,
															correlated = True,
															r_0 = 20,
															B = B,
															f = 10)
			mean_vec[ i, int(B / 5) ] = np.mean(g_bar)
			sd_vec[ i, int(B / 5) ] = np.std(g_bar)

	mean_vec = np.mean(mean_vec, axis = 0)
	sd_vec = np.mean(sd_vec, axis = 0)

	f = plt.figure(7)

	plt.plot( np.linspace(0,20,5) , mean_vec / nS, 'x-', color = "blue", label = r'$\mu$')
	plt.plot( np.linspace(0,20,5) , sd_vec / nS, 'x-', color = "red", label = r'$\sigma$')
	plt.title(r'\textbf{Distribution of the steady-state synaptic strengths}', fontsize=18)
	plt.xlabel(r'B (Hz)', fontsize=16)
	plt.ylabel(r'Synaptic strengths (nS)', fontsize=16)
	plt.tick_params(labelcolor='black', labelsize='16', width=1)
	plt.legend(prop={"size":16})
	plt.show()
