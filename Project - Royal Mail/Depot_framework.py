import math
import numpy as np
import matplotlib.pyplot as plt

# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#########################################

def Dies_annual_cost(total_miles_Dies, area, price_litre):

	litres_anum = total_miles_Dies * fuel_efficiency(area) # total litres burnt over a year in a designated area
	Dies_cost_anum = litres_anum * price_litre

	return Dies_cost_anum # cost of diesel fuel for a year

def maintenance_cost(car_type):
	if car_type == 'diesel':
		return 1400 # £
	elif car_type == 'electric':
		return 1100 # £
	else:
		print('diesel or electric')

def CO_2_reduction(C0_2_emit, total_miles_E, total_miles_Dies, C0_2_emit_E):

	C0_2_anum = C0_2_emit_E * total_miles_E + total_miles_Dies * C0_2_emit
	
	return C0_2_anum

def depot_cars(depot_size):
	if depot_size == 'small': # 8% of the depots
		return 18
	elif depot_size == 'medium': # 91% of the depots
		return 32 
	elif depot_size == 'large': # 1% of the depots
		return 120 # total cars in the depot
	elif depot_size == 'RM':
		return 37300 # sum of the depots
	else:
		print('small, medium, large, or RM')

def depot(depot_size):
	
	RM_miles_anum = 879894412
	RM_total_cars = depot_cars('RM')

	depot_miles = (RM_miles_anum / RM_total_cars) * depot_cars(depot_size)

	depot_budget = (50000000 / RM_total_cars) * depot_cars(depot_size)

	return depot_miles, depot_budget

def fuel_efficiency(area):
	# Average fuel efficiency of Peugeot Experts, Vauxhall Vivaros and Ford Transit Customs.
	# Assume an equal split for the 3 models.
	# Equated from miles per gallon to litres per mile

	urban_efficiency = ( (1/(39*0.22)) + (1/(35*0.22)) + (1/(35*0.22)) ) / 3
	rural_efficiency = ( (1/(45*0.22)) + (1/(42*0.22)) + (1/(42*0.22)) ) / 3

	if area == 'urban':
		return urban_efficiency
	elif area == 'rural':
		return  rural_efficiency # litre of fuel per mile
	elif area == 'RM':
		weighted_efficiency = 0.07*rural_efficiency + 0.93*urban_efficiency
		return  weighted_efficiency
	else:
		print('urban, rural, or RM')

def fleet_conversion_0FC(num_E_cars):

	cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	Energy_cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	RC = math.ceil( math.ceil(num_E_cars * 0.15) / 3 )
	count_RC = 1
	FC = 0
	Charger_cost = np.zeros(num_E_cars+1)

	for i in range(1, num_E_cars + 1):

		if i < (RC * 7 + 1):

			if ( i - (FC + 1) )%7 == 0 and count_RC <= RC:
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312
				Charger_cost[i] = 15000
				count_RC += 1
			else:	
				Energy_cars[i] = ( i * 41.4 * 0.087 ) * 312

		elif i < (num_E_cars - math.ceil(num_E_cars * 0.15) + 1):
			Energy_cars[i] = (RC * 7 * 41.4 * 0.087) * 312 + ((i - (RC * 7)) * 41.4 * 0.153) * 312


		else:
			Energy_cars[i] = (RC * 7 * 41.4 * 0.087) * 312 + ((((num_E_cars-math.ceil(num_E_cars*0.15)) - RC * 7) + (i-(num_E_cars-math.ceil(num_E_cars*0.15)))*2) * 41.4 * 0.153) * 312

	return [cars, Energy_cars, Charger_cost]

def fleet_conversion_10FC(num_E_cars):
	num_E_cars = math.ceil(num_E_cars * 1.15)
	cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	Energy_cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	Charger_cost = np.zeros(num_E_cars+1)

	RC = 0
	FC = math.ceil( num_E_cars / 4 )


	for i in range(1, len(cars)):

		if i <= FC:
			Energy_cars[i] = (i * 41.4 * 0.087) * 312
			Charger_cost[i] =  1000

		else:
			Energy_cars[i] = ((FC * 41.4 * 0.087) + (i - FC) * 41.4 * 0.153*(17/18)) * 312 + ((i - FC) * 41.4 * 0.087*(1/18)) * 312
			
	return [cars, Energy_cars, Charger_cost]

def fleet_conversion_18FC(num_E_cars):

	cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	Energy_cars = np.linspace(0, num_E_cars, num_E_cars + 1)
	RC = math.ceil( math.ceil(num_E_cars * 0.15) / 3 )
	FC = num_E_cars - (RC * 7)
	Charger_cost = np.zeros(num_E_cars+1)

	Chargers = np.zeros((num_E_cars+1, 2))

	for i in range(1, num_E_cars+1):

		if i < 4:

			if i < ( num_E_cars - math.ceil( num_E_cars * 0.15 ) ):

				Energy_cars[i] = 0 #( i * 41.4 * 0.087 ) * 312

				if i < ( FC + 1 ):
					Charger_cost[i] =  1000
					Chargers[i, 0] = 1
				elif ( i - (FC + 1) )%7 == 0:
					Charger_cost[i] = 15000
					Chargers[i, 1] = 1

		else:

			if i < ( num_E_cars - math.ceil( num_E_cars * 0.15 ) ):

				Energy_cars[i] = ( (i-3) * 41.4 * 0.087 ) * 312

				if i < ( FC + 1 ):
					Charger_cost[i] =  1000
					Chargers[i, 0] = 1
				elif ( i - (FC + 1) )%7 == 0:
					Charger_cost[i] = 15000
					Chargers[i, 1] = 1

			else:

				if ( i - (FC + 1) )%7 == 0:
					Charger_cost[i] = 15000
					Chargers[i, 1] = 1
					Energy_cars[i] = ( (i-3) * 41.4 * 0.087 ) * 312
				else:
					Energy_cars[i] = ((i-3) * 41.4 * 0.087) * 312 + (((i-3) - ( num_E_cars - math.ceil(num_E_cars * 0.15) )) * 41.4 * 0.153) * 312
			
	return [cars, Energy_cars, Charger_cost, Chargers]

def etotal(elec, ch, depot_size, num_E_cars, price_E_car, inv_anum, count):
	spendings = np.zeros(depot_cars(depot_size))
	for i in range(depot_cars(depot_size)):
		if num_E_cars + i + 1 > depot_cars(depot_size):
			spendings[i] = i * price_E_car + elec[num_E_cars + i] + np.sum( ch[num_E_cars + 1 : depot_cars(depot_size)] + ch[-1])
			num_E_cars_bought = i
			E_car_ch_elec = spendings[i]
			E_ch_elec = spendings[i] - (i) * price_E_car
			break
		else:
			if count < 6:
				if i == 0 :
					spendings[i] = elec[num_E_cars] + ch[ num_E_cars + 1 ] + 36652
					print('1 solar bought')
					count += 1
				else:
					spendings[i] = i * price_E_car + elec[num_E_cars + i] + np.sum( ch[num_E_cars + 1 : num_E_cars + i + 1] ) + 36652*i
					count += i
					print('{}, solar bought'.format(i))
			else:
				if i == 0 :
					spendings[i] = elec[num_E_cars] + ch[ num_E_cars + 1 ]
				else:
					spendings[i] = i * price_E_car + elec[num_E_cars + i] + np.sum( ch[num_E_cars + 1 : num_E_cars + i + 1] )

		if spendings[i] > inv_anum:
			num_E_cars_bought = i - 1
			E_car_ch_elec = spendings[i - 1]
			E_ch_elec = spendings[i - 1] - (i - 1) * price_E_car
			break

	return num_E_cars_bought, E_car_ch_elec, E_ch_elec, count

def main(budget, savings, total_miles_E, num_E_cars, price_E_car, 
		 price_Dies_car, total_miles, area, price_litre, total_miles_Dies, num_Dies_cars, count):

	# Update annual investment budget
	inv_anum = budget + savings

	# Cost breakdown for the electric cars
	cost_E_cars_maintenance = num_E_cars * maintenance_cost('electric')

	elec0 =  fleet_conversion_0FC( depot_cars(depot_size) )[1]
	ch0 =  fleet_conversion_0FC( depot_cars(depot_size) )[2]
	elec18 =  fleet_conversion_18FC( depot_cars(depot_size) )[1]
	ch18 =  fleet_conversion_18FC( depot_cars(depot_size) )[2]
	elec10 =  fleet_conversion_10FC( depot_cars(depot_size) )[1]
	ch10 =  fleet_conversion_10FC( depot_cars(depot_size) )[2]

	elec = elec18
	ch = ch18

	num_E_cars_bought, E_car_ch_elec, E_ch_elec, count = etotal(elec, ch, depot_size, num_E_cars, price_E_car, inv_anum, count)

	# _, _, E_ch_elec0, count = etotal(elec0, ch0, depot_size, num_E_cars, price_E_car, inv_anum, count)
	# _, _, E_ch_elec10, count = etotal(elec10, ch10, depot_size, num_E_cars, price_E_car, inv_anum, count)


	if num_E_cars_bought > num_Dies_cars:
		num_E_cars_bought = num_Dies_cars

	# Savings breakdown for the Diesel cars
	num_Dies_cars_sold = num_E_cars_bought
	savings_Dies_car_sold = num_Dies_cars_sold * price_Dies_car
	savings_Diesel = Dies_annual_cost(total_miles_E, area, price_litre)
	savings_Dies_cars_maintenance = num_E_cars * maintenance_cost('diesel')

	# Total cost + saving breakdown
	total_E_cost = cost_E_cars_maintenance + E_car_ch_elec #cost_E_cars_bought + cost_chargers_bought + cost_electricity + cost_E_cars_maintenance
	total_Dies_save = savings_Dies_car_sold + savings_Diesel + savings_Dies_cars_maintenance

	# Outputs
	savings = (inv_anum - total_E_cost) + total_Dies_save

	num_E_cars = num_E_cars + num_E_cars_bought
	num_Dies_cars = num_Dies_cars - num_Dies_cars_sold
	total_miles_E = total_miles * (num_E_cars/(num_E_cars + num_Dies_cars))
	total_miles_Dies = total_miles * (num_Dies_cars/(num_E_cars + num_Dies_cars))

	return savings, num_E_cars, num_Dies_cars, total_miles_E, total_miles_Dies, inv_anum, E_car_ch_elec, E_ch_elec, count


def annual_results(area, years, savings, total_miles_E, num_E_cars, total_miles_Dies, num_Dies_cars, C0_2_emit, budget):
	year_plot = np.linspace(0, years-1, int(years))

	inv_anum_vec = np.zeros(years)
	savings_vec = np.zeros(years)

	co2 = np.linspace(0, years-1, int(years))
	C0_2_emit_E = np.zeros(18)
	# C0_2_emit_E[0:16] = np.linspace(148/1e6, 41/1e6, 16)
	# C0_2_emit_E[15:19] = 41/1e6
	co2[0] = CO_2_reduction(C0_2_emit, total_miles_E, total_miles_Dies, C0_2_emit_E[0])

	cars = np.zeros((years, 2))
	cars[0,:] = [num_Dies_cars, num_E_cars]

	plto = np.zeros(years)

	print('\nInitial position:')
	print('Number of electric cars =',num_E_cars)
	print('Number of diesel cars =',num_Dies_cars)
	# print('Current annual C0_2 emission: {:,.2f} tonnes per year'.format(co2[0]))

	count = 0

	for i in range(years):
		savings, num_E_cars, num_Dies_cars, total_miles_E, total_miles_Dies, inv_anum, E_car_ch_elec, E_ch_elec, count = main(budget = budget,
																											savings = savings,
																											total_miles_E = total_miles_E,
																											num_E_cars = num_E_cars,
																											price_E_car = 40000*0.90,
																											price_Dies_car = 2000,
																											total_miles = depot(depot_size)[0],
																											area = area,
																											price_litre = 1.3,
																											total_miles_Dies = total_miles_Dies,
																											num_Dies_cars = num_Dies_cars,
																											count = count)
		print('\nEnd of Year', i+1)
		print('inv_anum = £{:,.2f}'.format(inv_anum))
		print('leftover = £{:,.2f}'.format(savings))
		print('Number of electric cars =',num_E_cars)
		print('Number of diesel cars =',num_Dies_cars)
		# print('Total reduction of annual C0_2 emission from initial position: {:,.2f} tonnes per year'.format(CO_2_reduction(C0_2_emit, total_miles_E, total_miles_Dies, C0_2_emit_E[i+1])))


		# inv_anum_vec[i+1] = inv_anum
		# savings_vec[i+1] = savings
		# co2[i+1] = CO_2_reduction(C0_2_emit, total_miles_E, total_miles_Dies, C0_2_emit_E[i+1])
		cars[i+1,:] = [num_Dies_cars, num_E_cars]
		plto[i+1] = E_ch_elec


		if num_Dies_cars <= 0:
			# inv_anum_vec = inv_anum_vec[0:i+2]/1000
			# savings_vec = savings_vec[0:i+2]/1000
			year_plot = year_plot[0:i+2]
			# co2 = co2[0:i+2]
			cars = cars[0:i+2]
			plto = plto[0:i+2]
			break

	return cars, plto

	# print('')

	# f = plt.figure(1)

	# plt.plot(np.linspace(0,year_plot[-1]+1,int(year_plot[-1]+2)), np.ones(len(year_plot)+1)*budget/1000, '--', color = 'black', label = r'Initial Annual Input')
	# plt.bar(year_plot, inv_anum_vec, alpha = 0.7)#, color = 'black')
	# plt.title(r'\textbf{Annual Net Investment}', fontsize=18)
	# plt.xlabel(r'Year', fontsize=16)
	# plt.xticks(np.linspace(0,17,18))
	# plt.ylabel(r'Investment (£ thousand)', fontsize=16)
	# plt.xlim([0, 18])
	# plt.ylim([0, 200])
	# plt.grid(True, linestyle='-.')
	# plt.legend(prop={"size":16}, loc = 'upper left')
	# plt.tick_params(labelcolor='black', labelsize='large', width=1)
	# plt.show()

	# f = plt.figure(2)

	# plt.plot(year_plot, savings_vec, 'x')
	# plt.title(r'\textbf{Annual savings after investment at the end of every year}', fontsize=18)
	# plt.xlabel(r'End of year', fontsize=16)
	# plt.xticks(np.linspace(0,years,int(years+1)))
	# plt.ylabel(r'Savings (£ thousand)', fontsize=16)
	# plt.xlim([0, int(year_plot[-1]+1)])
	# plt.ylim([0, int(max(savings_vec)+20)])
	# plt.grid(True, linestyle='-.')
	# plt.tick_params(labelcolor='black', labelsize='large', width=1)
	# plt.show()

	# f = plt.figure(3)
 
	# plt.plot(year_plot, np.array(co2)/1000, '-.', alpha = 0.7, linewidth = 3)#, color = 'red', label = r'MCE')
	# plt.title(r'\textbf{Royal Mail CO$_{2}$ Emissions}', fontsize=18)
	# plt.xlabel(r'Year', fontsize=16)
	# plt.xticks(np.linspace(0,17,18))
	# plt.ylabel(r'CO$_{2}$ emissions (kilo-tonnes)', fontsize=16)
	# plt.xlim([0, 17])
	# plt.ylim([0, 185])
	# plt.grid(True, linestyle='-.')
	# # plt.legend(prop={"size":16}, loc = 'upper right')
	# plt.tick_params(labelcolor='black', labelsize='large', width=1)
	# plt.show()

	# f = plt.figure(4)
 
	# plt.scatter(year_plot, cars[:,0], s = 50, marker = 'x', c = 'red', label = 'Diesel Cars')
	# plt.scatter(year_plot, cars[:,1], s = 50, marker = 'x', c = 'green', label = 'Electric Cars')
	# plt.title(r'\textbf{Fleet Distribution}', fontsize=18)
	# plt.xlabel(r'Year', fontsize=16)
	# plt.xticks(np.linspace(0,years,int(years+1)))
	# plt.ylabel(r'Number of Cars', fontsize=16)
	# plt.xlim([0, 17])
	# plt.ylim([0, 35])
	# plt.grid(True, linestyle='-.')
	# plt.legend(prop={"size":14})
	# plt.tick_params(labelcolor='black', labelsize='large', width=1)
	# plt.show()


depot_size = 'medium'
cars, plto = annual_results(area = 'urban', years = 40, savings = 0, total_miles_E = 0, num_E_cars = 0, total_miles_Dies = depot(depot_size)[0],
				num_Dies_cars = depot_cars(depot_size), C0_2_emit = 185/1000000, budget = depot(depot_size)[1])

print(plto)

plot_of_18 = [    0    ,  2123.7616 , 3247.5232 , 4371.2848  ,5495.0464  ,8742.5696,
  8866.3312 ,12113.8544, 14361.3776 ,14485.1392, 17732.6624 ,22103.9472,
 21227.7088, 38598.9936, 26970.2784, 44217.8016, 36541.6272, 45841.7232]

# plot_of_0 = [    0   ,  15000.  ,   1123.7616 , 2247.5232 , 3371.2848 , 5618.808,
#   6742.5696 , 7866.3312 ,25113.8544, 12361.3776, 14608.9008,17708.9328,
#  21661.4736 ,25614.0144, 31542.8256, 35495.3664,41424.1776 ,53281.8,
#  61186.8816]

# plot_of_10 = [    0,      2123.7616,  3247.5232,  4371.2848,  5495.0464,  8742.5696,
#   8866.3312, 12113.8544, 12237.616,  15095.4336, 18953.2512, 22811.0688,
#  26668.8864, 30526.704,  34384.5216, 38242.3392, 42100.1568, 47886.8832,
#  51744.7008, 57531.4272, 63318.1536]


f = plt.figure(5)

plt.bar(np.linspace(0,17,18), np.array(plot_of_18)/1000,  alpha = 0.5, color = 'red', label = r'MEE')
plt.bar(np.linspace(0,19,20), np.array(plto)/1000,  alpha = 0.5, color = 'blue', label = r'MEE with Solar')
# plt.bar(np.linspace(0,18,19), np.array(plot_of_0)/1000,  alpha = 0.5, color = 'blue', label = r'MCE')
# plt.bar(np.linspace(0,20,21), np.array(plot_of_10)/1000,  alpha = 0.5, color = 'green', label = r'MGP')
plt.title(r'\textbf{Investment on Chargers and Electricity -\\ \indent Medium Sized Depot}', fontsize=18)
plt.xlabel(r'Time (Years)', fontsize=16)
plt.xticks(np.linspace(0,19,20))
plt.ylabel(r'Investment (£ thousand)', fontsize=16)
plt.xlim([0, 20])
plt.ylim([0, 65])
plt.grid(True, linestyle='-.')
plt.tick_params(labelcolor='black', labelsize='large', width=1)
plt.legend(prop={"size":16})
plt.show()



# [_, _, _, Chargers] = fleet_conversion_18FC(32)

# Chargers = np.cumsum(Chargers, axis = 0)

# Chargers_plot = np.zeros((len(cars), 2))

# cars = cars.astype(int)

# for i in range(len(cars)):
# 	Chargers_plot[i,:] = Chargers[cars[i,1],:]

# year_plot = np.linspace(0, 17, 18)

# f = plt.figure(6)
 
# plt.scatter(year_plot, Chargers_plot[:,0], s = 50, marker = 'x', c = 'blue', label = r'Fast Chargers')
# plt.scatter(year_plot, Chargers_plot[:,1], s = 50, marker = 'x', c = 'red', label = r'Rapid Chargers')
# plt.title(r'\textbf{Charger Distribution}', fontsize=18)
# plt.xlabel(r'Year', fontsize=16)
# plt.xticks(np.linspace(0,17,18))
# plt.yticks(np.linspace(0,20,11))
# plt.ylabel(r'Number of Chargers', fontsize=16)
# plt.xlim([0, 17])
# plt.ylim([0, 20])
# plt.grid(True, linestyle='-.')
# plt.legend(prop={"size":16}, loc = 'upper left')
# plt.tick_params(labelcolor='black', labelsize='large', width=1)
# plt.show()


