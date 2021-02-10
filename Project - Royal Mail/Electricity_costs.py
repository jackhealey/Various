import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
#### medium delivery centre ###

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def chargers(RC,FC,t):
	time_energy = np.zeros((24*(t)+1,2))
	time_energy_T = np.zeros((24*(t)+1,2))
	time_energy[:,0] = np.linspace(0,24,24*t+1)
	time_energy_T[:,0] = np.linspace(0,24,24*t+1)
	if RC == 2:
			
		#Corner case: minimum spent on chargers. Charging between shifts needs two rapid chargers
		if FC == 0:
			time_energy[int(t*(6))    +1:int(t*(12.5)) +1,1] = 0 # 3 fast chargers in use
			time_energy[int(t*(12.5)) +1:int(t*(14+1/3))   +1,1] = 50*2			# rapid charge for doubler shift then morning shift vans
			time_energy[int(t*(14+1/3)) +1:int(t*(15.25))   +1,1] = 50*1	
			time_energy[int(t*(15.25)) +1:int(t*(16.25))   +1,1] = 0
			time_energy[int(t*(16.25)) +1:int(t*(24))   +1,1] = 50*2
			time_energy[int(t*(0))      :int(t*(6.916))      ,1] = 50*2
			time_energy_T[0:-1,1] = time_energy[0:-1,1]*(41.4/45.83)
		#Corner case: minimum spent on electricity. Linear programming 
		if FC == 18:
			time_energy[int(t*(6))    +1:int(t*(6.416)) +1,1] = RC*50
			time_energy[int(t*(6.416)) +1:int(t*(12.5))    +1,1] = 0
			time_energy[int(t*(12.5)) +1:int(t*(14+1/3)) +1,1] = 50*RC
			time_energy[int(t*(14+1/3)) +1:int(t*(15.25))+1,1] = 50*1
			time_energy[int(t*(15.25))+1:int(t*(24))   +1,1] = 0
			time_energy[int(t*(0))      :int(t*(6))    +1,1] = 7*18 + 2*50
		
			#time_energy_T = time_energy
			time_energy_T[int(t*(6))    +1:int(t*(6.416)) +1,1] = time_energy[int(t*(6))    +1:int(t*(6.416)) +1,1] *(41.4/45.83)
			time_energy_T[int(t*(12.5)) +1:int(t*(15.25))+1,1] = time_energy[int(t*(12.5)) +1:int(t*(15.25))+1,1] * (41.4/45.83)
			time_energy_T[int(t*(0))      :int(t*(6))    +1,1] = time_energy[int(t*(0))      :int(t*(6))    +1,1] * (1207/1272)
		
		if FC == 10:
			time_energy[int(t*6) + 1	:	int(t*24) +1,1] = 7*9
			time_energy[int(0)      :	int(t*6) +1,1] = 7*10
			
			
	return[time_energy,time_energy_T]	

def fleet_conversion_0FC(Num_Ecar):

	cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)
	Energy_cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)
	RC = math.ceil( math.ceil(Num_Ecar * 0.15) / 3 )
	count_RC = 1
	FC = 0

	for i in range(1, Num_Ecar + 1):

		if i < (RC * 7 + 1):

			if ( i - (FC + 1) )%7 == 0 and count_RC <= RC:
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312 + 15000
				count_RC += 1
			else:	
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312

		elif i < (Num_Ecar - math.ceil(Num_Ecar * 0.15) + 1):
			Energy_cars[i] = (RC * 7 * 41.4 * 0.087) * 312 + ((i - (RC * 7)) * 41.4 * 0.153) * 312


		else:
			Energy_cars[i] = (RC * 7 * 41.4 * 0.087) * 312 + ((((Num_Ecar-math.ceil(Num_Ecar*0.15)) - RC * 7) + (i-(Num_Ecar-math.ceil(Num_Ecar*0.15)))*2) * 41.4 * 0.153) * 312
			
	return [cars, Energy_cars]
	
def fleet_conversion_18FC(Num_Ecar):

	cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)
	Energy_cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)
	RC = math.ceil( math.ceil(Num_Ecar * 0.15) / 3 )
	FC = Num_Ecar - (RC * 7)

	for i in range(1, Num_Ecar + 1):

		if i < ( Num_Ecar - math.ceil( Num_Ecar * 0.15 ) ):

			if i < ( FC + 1 ):
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312 + 1000
			elif ( i - (FC + 1) )%7 == 0:
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312 + 15000
			else:	
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312

		else:
			if ( i - (FC + 1) )%7 == 0:
				Energy_cars[i] = (i * 41.4 * 0.087) * 312 + ((i - ( Num_Ecar - math.ceil(Num_Ecar * 0.15) )) * 41.4 * 0.153) * 312 + 15000
			else:
				Energy_cars[i] = (i * 41.4 * 0.087) * 312 + ((i - ( Num_Ecar - math.ceil(Num_Ecar * 0.15) )) * 41.4 * 0.153) * 312
			
	return [cars, Energy_cars]

def fleet_conversion_10FC(Num_Ecar):
	Num_Ecar = math.ceil(Num_Ecar * 1.15)
	cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)
	Energy_cars = np.linspace(0, Num_Ecar, Num_Ecar + 1)

	RC = 0
	FC = math.ceil( Num_Ecar / 4 )


	for i in range(1, len(cars)):

		if i <= FC:
			Energy_cars[i] = (i * 41.4 * 0.087) * 312 + 1000

		else:
			Energy_cars[i] = ((FC * 41.4 * 0.087) + (i - FC) * 41.4 * 0.153*(17/18)) * 312 + ((i - FC) * 41.4 * 0.087*(1/18)) * 312
			
	return [cars, Energy_cars]


# [cars_0FC, Energy_cars_0FC] = 	fleet_conversion_0FC(32)
# [cars_18FC, Energy_cars_18FC] = fleet_conversion_18FC(32)
# [cars_10FC, Energy_cars_10FC] = fleet_conversion_10FC(32)

# f = plt.figure(1)

# plt.bar(cars_10FC, Energy_cars_10FC, alpha = 0.5, color = 'green',label = r'RC = 0, FC = 10, plus 15\% EVs')
# plt.bar(cars_18FC, Energy_cars_18FC, alpha = 0.5, color = 'red', label = r'RC = 2, FC = 18')
# plt.bar(cars_0FC, Energy_cars_0FC, alpha = 0.5, color = 'blue',  label = r'RC = 2, FC = 0')
# plt.title(r'\textbf{Money spent on Charging and Chargers}', fontsize=18)
# plt.xlabel(r'Number of Electric Vans', fontsize=16)
# plt.ylabel(r'Pounds Spent on Charging Fleet', fontsize=16)
# plt.tick_params(labelcolor='black', labelsize='16', width=1)
# plt.legend(prop={"size":16})
# plt.show()

time_0FC  = chargers(2,0,100)[0][:,0]
energy_0FC= chargers(2,0,100)[0][:,1]
time_18FC  = chargers(2,18,100)[0][:,0]
energy_18FC= chargers(2,18,100)[0][:,1]
time_10FC  = chargers(2,10,100)[0][:,0]
energy_10FC= chargers(2,10,100)[0][:,1]



f = plt.figure(2)

plt.plot(time_18FC , energy_18FC, color = "red", label = r'MEE', linewidth = 5)
# plt.plot(time_18FC[602:640] , energy_18FC[602:640], color = "red", linewidth = 4)
# plt.plot(time_18FC[692: 1625] , energy_18FC[692: 1625], color = "red", linewidth = 4)
plt.plot(time_0FC , energy_0FC, color = "blue", label = r'MCE')
plt.plot(time_10FC , energy_10FC, color = "green", label = r'MGP')
plt.title(r'\textbf{Power Drawn from the National Grid}', fontsize=18)
plt.xlabel(r'Time of Day (Hours)', fontsize=16)
plt.ylabel(r'Power (kW)', fontsize=16)
plt.xlim([0, 24])
plt.xticks(np.linspace(0,24,13))
plt.grid(True, linestyle='-.')
plt.tick_params(labelcolor='black', labelsize='14', width=1)
plt.legend(prop={"size":16})
plt.show()


