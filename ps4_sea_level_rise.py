from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import PIL, PIL.Image
import scipy

import scipy.stats as st
# from scipy.interpolate import interp1d

class floodmapper:
    """
    Create a floodmapper instance, for a map in a specified bounding box and a particular mass gis dbf file.
    Parameters are:
        tl - (lat,lon) pair specifying the upper left corner of the map data to load

        br - (lat,lon) pair specifying the bottom right corner of the map data to load 

        z - Zoom factor of the map data to load
        
        dbf_path - Path to the dbf file listing properties to load
        
        load - Boolean specifying whether or not to download the map data from mapbox.  If false, mapbox files must 
        have been previously loaded.  If using this as a 6.0001 problem set, you should have received pre-rendered map tiles
        such that it is OK to pass false to this when using the specific parameters in the __main__ code block below.
    """
    def __init__(self, tl, br, z, dbf_path, load):
        self.mtl = maptileloader.maptileloader(tl, br, z)
        dbf = massdbfloader.massdbfloader(dbf_path)
        if (load):
            self.mtl.download_tiles()
        self.pts = dbf.get_points_in_box(tl,br)
        self.ul, self.br = self.mtl.get_tile_extents()
        self.elevs = self.mtl.get_elevation_array()

    """
    Return a rendering as a PIL image of the map where properties below elev are highlighted
    """
    def get_image_for_elev(self, elev):
        fnt = ImageFont.truetype("Arial.ttf", 80)
        im = self.mtl.get_satellite_image()
        draw = ImageDraw.Draw(im)
        draw.text((10,10), f"{elev} meters", font=fnt, fill=(255,255,255,128))
        for name, lat, lon, val in self.pts:
            # print(name)
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            # print(x,y)
            # print(e[x,y])
            el = int(self.elevs[y,x]*15)
            #print(e[y,x])
            c = f"rgb(0,{el},200)"
            if (self.elevs[y,x] < elev):
                c = f"rgb(255,0,0)"
            draw.ellipse((x-3,y-3,x+3,y+3), PIL.ImageColor.getrgb(c))
        return im

    """
    Return an array of (property name, lat, lon, elevation (m), value (USD)) tuples where properties
    are below the specified elev.
    """
    def get_properties_below_elev(self, elev):
        out = []
        for name, lat, lon, val in self.pts:
            x = int((((lon - self.ul[0]) / (self.br[0] - self.ul[0]))) * self.elevs.shape[1])
            y = int((((lat - self.ul[1]) / (self.br[1] - self.ul[1]))) * self.elevs.shape[0])
            if (self.elevs[y,x] < elev):
                out.append((name,lat,lon, self.elevs[y,x], val))

        return out



#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)


def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)


def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())


###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""

    all_years = load_slc_data()[0]
    lst_of_means = []
    lst_of_upper = []
    lst_of_lower = []

    #the array to return with year, mean, 2.5 and 97.5 percentile, std
    yearly_temp = [[x]+4*[0] for x in range(2020,2100+1)]
    
    #loop through each year
    for i in range(2020, 2100+1):
        inx = i-2020 #index of array 
        
        #if slr is in data given find data and make lower and upper variables 
        if i in all_years: 
            find_inx = list(load_slc_data()[0]).index(i)
            lower = float(load_slc_data()[1][find_inx])
            upper = float(load_slc_data()[2][find_inx])
        #if slr is not in data need to use interp function 
        else: 
            lower = float(interp(i, all_years, load_slc_data()[1]))
            upper = float(interp(i, all_years, load_slc_data()[2]))
        
        mean = (lower + upper)/2

        yearly_temp[inx][1] = mean
        yearly_temp[inx][2] = lower
        yearly_temp[inx][3] = upper
        yearly_temp[inx][4] = float(calculate_std(np.array([upper]), np.array([mean]))) #std 
        
        #y-axis for plotting 
        lst_of_means.append(mean)
        lst_of_upper.append(upper)
        lst_of_lower.append(lower)
            
    
        
    if show_plot:
        #plot slr of each year
        plt.title(("(expected results)"))
        plt.xlabel('Year')
        plt.ylabel('Projected annual mean water level (ft)')
        plt.xlim(2020,2100)
        
        plt.plot([x for x in range(2020,2100+1)], lst_of_upper, '--', label='Upper')
        plt.plot(range(2020,2100+1), lst_of_lower, '--', label='Lower')
        plt.plot([x for x in range(2020,2100+1)], lst_of_means, '-', label='Mean')
        plt.legend() 
        
        plt.figtext(.5, -0.04, "Figure 2.la Time-series of projected annual average water level and 95% confidence interval.", 
                    wrap=True, horizontalalignment='center', fontstyle='italic', color='mediumblue')
        
    return np.array(yearly_temp)


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    #use mean and standard reduction to find possible slr 
    sim_values = []
    array_inx = year-2020 
    for i in range(num):
        sim_values.append(np.random.normal(data[array_inx, 1], data[array_inx, 4]))
        
    return np.array(sim_values)
    
    
def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    lst_of_means = []
    lst_of_lower = []
    lst_of_upper = []
    all_years_sims = []
    
    for array in data: 
        lst_of_means.append(array[1])
        lst_of_lower.append(array[2])
        lst_of_upper.append(array[3])
    
    #simulation for each year
    for yr in range(2020,2101):
        all_years_sims.append(simulate_year(data, yr, 500)) 
    
        
    plt.title(("(expected results)"))
    plt.xlabel('Year')
    plt.ylabel('Projected annual mean water level (ft)')
    plt.xlim(2020,2100)
    plt.ylim(0,14)
    
    plt.plot([x for x in range(2020,2100+1)], lst_of_upper, '--', label='Upper')
    plt.plot(range(2020,2100+1), lst_of_lower, '--', label='Lower')
    plt.plot([x for x in range(2020,2100+1)], lst_of_means, '-', label='Mean')
    for yr in range(2020,2101):
        yr_sim = all_years_sims[yr-2020]
        for sim in yr_sim:
            plt.scatter(yr, sim, color='gray', s=0.1, linewidths=0)

    plt.legend() 
    
    
    

##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    water_lvls = []
    for yr in range(2020, 2101): 
        water_lvls.append(float(simulate_year(data, yr, 1)))
    return water_lvls

def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""   
    
    damage_costs = []
    #made sure to include damages costs if slr < 5 
    table_slr = [0, 1, 2, 3 , 4]
    table_pcng = [0]*5
    for row in water_level_loss_no_prevention: 
        table_slr.append(row[0])
        table_pcng.append(row[1])
    #used to interpolate water_level to property damage     
    interp_water = scipy.interpolate.interp1d(table_slr, table_pcng)
    
    for yr in range(2020,2101): 
        slr = water_level_list[yr-2020]
        #if slr is in given table can find corresponding cost 
        if slr in table_slr: 
            inx = table_slr.index(slr)
            damage_pctg = table_pcng[inx]*.01
            
        else: 
            damage_pctg = interp_water(slr)*.01
        
        if slr <= 5: 
            damage_costs.append(0)
        elif 5 < slr < 10: 
            damage_costs.append(house_value*damage_pctg/1000)
        else: 
            damage_costs.append(house_value/1000)
            
    return damage_costs 
            
            
            


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    damage_costs = []
    flood_prevention = False 
    
    table_slr_no_pr = [0, 1, 2, 3 , 4]
    table_pcng_no_pr = [0]*5
    
    table_slr_pr = [0,1,2,3,4]
    table_pcng_pr = [0]*5
    
    for row in water_level_loss_no_prevention: 
        table_slr_no_pr.append(row[0])
        table_pcng_no_pr.append(row[1])
    
    for row in water_level_loss_with_prevention: 
        table_slr_pr.append(row[0])
        table_pcng_pr.append(row[1])
    
    
    #used to interpolate water_level to property damage  
    
    for yr in range(2020,2101): 
        if flood_prevention: 
            table_slr = table_slr_pr
            table_pcng = table_pcng_pr
        else: 
            table_slr = table_slr_no_pr
            table_pcng = table_pcng_no_pr
        
        interp_water = scipy.interpolate.interp1d(table_slr, table_pcng)
        
        slr = water_level_list[yr-2020]

        if slr in table_slr: 
            inx = table_slr.index(slr)
            damage_pctg = table_pcng[inx]*.01
        else: 
            damage_pctg = interp_water(slr)*.01
        
        if slr <= 5: 
            current_cost = 0 
        elif 5 < slr < 10: 
            current_cost = house_value*damage_pctg
        else: 
            current_cost = house_value 
  
        if current_cost > cost_threshold: 
            flood_prevention = True 
        # elif flood_prevention: 
        #     flood_prevention = False
       
        damage_costs.append(current_cost/1000)
        
    return damage_costs 



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    
    damage_costs = []
    table_slr = [0, 1, 2, 3 , 4]
    table_pcng = [0]*5
    for row in water_level_loss_with_prevention: 
        table_slr.append(row[0])
        table_pcng.append(row[1])
    #used to interpolate water_level to property damage     
    interp_water = scipy.interpolate.interp1d(table_slr, table_pcng)
    
    for yr in range(2020,2101): 
        slr = water_level_list[yr-2020]

        if slr in table_slr: 
            inx = table_slr.index(slr)
            damage_pctg = table_pcng[inx]*.01
        else: 
            damage_pctg = interp_water(slr)*.01
        
        if slr <= 5: 
            damage_costs.append(0)
        elif 5 < slr < 10: 
            damage_costs.append(house_value*damage_pctg/1000)
        else: 
            damage_costs.append(house_value/1000)
            
    return damage_costs 


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    
    plt.title(("Average Annual Damage Cost "))
    plt.xlabel('Year')
    plt.ylabel('Estimated Damage Cost ($K)')
    plt.xlim(2020,2100)
    plt.ylim(0,400)
    
    all_repairs_cost = []
    all_wait_repairs_cost = []
    all_prepared_cost = []

    
    for i in range(501): 
        all_years_slr = []
        for yr in range(2020, 2101):
            slr = 11 
            while slr > 10: 
                slr = float(simulate_year(data, yr, 1))
            all_years_slr.append(slr)

        
        repairs_cost = repair_only(all_years_slr, water_level_loss_no_prevention, house_value)
        wait_repairs_cost = wait_a_bit(all_years_slr, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
        prepared_cost = prepare_immediately(all_years_slr, water_level_loss_with_prevention, house_value)
        
        all_repairs_cost.append(repairs_cost)
        all_wait_repairs_cost.append(wait_repairs_cost)
        all_prepared_cost.append(prepared_cost)
        for yr in range(2020,2101):
            plt.scatter(yr, repairs_cost[yr-2020], color='red', s=0.1, linewidths=0)
            # all_repairs_cost[yr-2020].append(repairs_cost[yr-2020])
            plt.scatter(yr, wait_repairs_cost[yr-2020], color='blue', s=0.1, linewidths=0)
            # all_wait_repairs_cost[yr-2020].append(r)
            plt.scatter(yr, prepared_cost[yr-2020], color='green', s=0.1, linewidths=0)
            

    
    
    array1 = np.array(all_repairs_cost)
    all_repairs_mean = np.mean(array1, axis=0)
    
    array2 = np.array(all_wait_repairs_cost)
    all_wait_mean = np.mean(array2, axis=0)
    
    array3 = np.array(all_prepared_cost)
    all_prepared_mean = np.mean(array3, axis=0)
    

    
    plt.plot([x for x in range(2020,2100+1)], list(all_repairs_mean), '-', label='Repair Only Scenario', color = 'red')
    plt.plot([x for x in range(2020,2100+1)], list(all_wait_mean), '-', label='Wait-a-bit Scenario', color = 'blue')
    plt.plot([x for x in range(2020,2100+1)], list(all_prepared_mean), '-', label='Prepare Immediately Scenario', color = 'green')
    plt.legend() 
    
        
    
    
    
    



if __name__ == '__main__':
    
    # Comment out the 'pass' statement below to run the lines below it
    pass 

    import maptileloader
    import massdbfloader

    # # Uncomment the following lines to plot generate plots
    data = predicted_sea_level_rise()
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    #plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
    
    # # Uncomment the following lines to visualize sea level rise over a map of Boston
    # tl = (42.3586798 +.04, - 71.1000466 - .065)
    # br = (42.3586798 -.02, - 71.1000466 + .065)
    # dbf = 'cambridge_2021.dbf'
    # fm = floodmapper(tl,br,14,dbf,False)

    # print("Getting properties below 5m")
    # pts = fm.get_properties_below_elev(5.0)
    # print(f"First one: {pts[0]}")
    
    # print("The next few steps may take a few seconds each.")

    # fig, ax = plt.subplots(figsize=(12,10), dpi=144)
    
    # ims=[]
    # print("Generating image frames for different elevations")
    # for el_cutoff in np.arange(0,15,.5):
    #     # print(el_cutoff)
    #     im = fm.get_image_for_elev(el_cutoff)
    #     im_plt = ax.imshow(im, animated=True)
    #     if el_cutoff == 0:
    #         ax.imshow(im)  # show an initial one first

    #     ims.append([im_plt])

    # print("Building animation")
    # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
    #                                 repeat_delay=1000)
    # print("Saving animation to animation.gif")
    # ani.save('animation.gif', fps=30)

    # print("Displaying Image")
    # plt.show()
