# Import libraries
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

############################   Question 1   ###############################

# Unit variables
Hz = 1.0
sec = 1.0
ms = 0.001

# Input variables
rate = 35.0 * Hz # firing rate (Hz)
big_t = 1000 * sec # total time

tau_refs = [0*ms, 5*ms] # refractory period (ms)
width_windows = [10*ms, 50*ms, 100*ms]

# Generate spike trains simulated using a Poisson process with a refractory period
def get_spike_train(rate,big_t,tau_ref):

    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []


    exp_rate=rate/(1-tau_ref*rate)

    spike_train=[]

    t=rnd.expovariate(exp_rate)

    while t < big_t:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)

    return spike_train

def fano_factor(spike_train, window_width, big_t):

	print("Window of width {} ms".format(int(window_width/ms)))
	num_windows = round(big_t/window_width)

	spike_count = np.histogram(spike_train, num_windows, (0, big_t))[0]
	
	fano_factor = np.var(spike_count) / np.mean(spike_count)
	print("Fano factor of the spike count: {}".format(fano_factor))

def coef_variation(spike_train):

    #Calculate time difference between successive spikes
    interspike_intervals = np.diff(spike_train)

    coef_var = np.std(interspike_intervals) / np.mean(interspike_intervals)
    print("Coefficient of variation for interspike intervals: {}".format(coef_var))

print("\nQuestion 1:\n")
for i in range(len(tau_refs)):
	print("Refractory period is F = {} ms\n".format(int(tau_refs[i]/ms)))
	spike_train_1 = get_spike_train(rate, big_t, tau_refs[i])

	for j in range(3):
		fano_factor(spike_train_1, width_windows[j], big_t)

	coef_variation(spike_train_1)
	print()

############################   Question 2   ###############################

def load_data(filename,T):

	data_array = [T(line.strip()) for line in open(filename, 'r')]

	return data_array

#spikes=[int(x) for x in load_data("rho.dat")]
spikes = load_data("rho.dat",int)

def binary_to_continuous(binary, sample_time):
    continous = []

    for time_point, signal in enumerate(binary):
    	if signal == 1:
        	continous.append(time_point * sample_time)

    return continous

spike_train_2 = binary_to_continuous(spikes, 2*ms)

big_t = len(spikes) * 2*ms

print("Question 2:\n")
for i in range(3):
	fano_factor(spike_train_2, width_windows[i], big_t)

coef_variation(spike_train_2)
print()

###########################   Question 3   ###############################

def autocorr(spikes, lims, sample_time):

	coefs = [np.corrcoef(  np.array([spikes[:-t], spikes[t:]])  )[1,0] for t in range(1, int((lims[1]/2)+1) )]

	corr_coef = np.concatenate([coefs[::-1], [None], coefs])

	intervals = np.arange(lims[0], lims[1]+1, sample_time)

	f = plt.figure(1)

	plt.plot(intervals, corr_coef)
	plt.title(r'\textbf{Autocorrelogram over the range -100 ms to +100 ms}', fontsize=13)
	plt.xlabel(r'Intervals (ms)', fontsize=14)
	plt.ylabel(r'Correlation coefficient', fontsize=14)
	plt.xlim([lims[0], lims[1]])
	plt.ylim([-0.1, 0.25])
	plt.grid(True, linestyle='-.')
	plt.tick_params(labelcolor='black', labelsize='small', width=1)
	f.savefig("Autocorrelogram.pdf", bbox_inches='tight')
	plt.show()

autocorr(spikes, [-100,100], 2)

###########################   Question 4   ###############################

#stimulus=[float(x) for x in load_data("stim.dat")]
stimulus=load_data("stim.dat",float)

def spike_triggered_avg(stimulus, spike_train, window, sample_time):

	time_intervals = np.arange(0, window+sample_time, sample_time)

	STA = []
	for time_int in time_intervals:
		stim_val = 0
		N = 0
		for spike_time in spike_train:
			index = int((spike_time/ms - time_int) / sample_time)
			if index >= 0:
				stim_val = stim_val + stimulus[index]
				N = N + 1

		STA.append(stim_val/N)

	f = plt.figure(2)
 
	plt.plot(time_intervals, STA)
	plt.title(r'\textbf{Spike triggered average plot over a 100 ms window}', fontsize=13)
	plt.xlabel(r'Time before spike (ms)', fontsize=14)
	plt.xticks(np.linspace(0, window, int((window/10)+1) ))
	plt.ylabel(r'Average stimulus', fontsize=14)
	plt.xlim([-10, window])
	plt.ylim([-5, 40])
	plt.grid(True, linestyle='-.')
	plt.axhline(linewidth=1, color='black')
	plt.axvline(linewidth=1, color='black')
	plt.tick_params(labelcolor='black', labelsize='small', width=1)
	f.savefig("Spike_Triggered_Average.pdf", bbox_inches='tight')
	plt.show()

spike_triggered_avg(stimulus, spike_train_2, 100, 2)
	
