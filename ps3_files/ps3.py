# -*- coding: utf-8 -*-
# Problem Set 3: Simulating robots
# Name: Jessica Jimenez 
# Collaborators (discussion):
# Time: 7 hours 

import math
import random
import matplotlib

from ps3_visualize import *
import pylab

# === Provided class Position, do NOT change
class Position(object):
    """
    A Position represents a location in a two-dimensional room, where
    coordinates are given by floats (x, y).
    """
    def __init__(self, x, y):
        """
        Initializes a position with coordinates (x, y).
        """
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_new_position(self, angle, speed):
        """
        Computes and returns the new Position after a single clock-tick has
        passed, with this object as the current position, and with the
        specified angle and speed.

        Does NOT test whether the returned position fits inside the room.

        angle: float representing angle in degrees, 0 <= angle < 360
        speed: positive float representing speed

        Returns: a Position object representing the new position.
        """
        old_x, old_y = self.get_x(), self.get_y()

        # Compute the change in position
        delta_y = speed * math.cos(math.radians(angle))
        delta_x = speed * math.sin(math.radians(angle))

        # Add that to the existing position
        new_x = old_x + delta_x
        new_y = old_y + delta_y

        return Position(new_x, new_y)

    def __str__(self):
        return "Position: " + str(math.floor(self.x)) + ", " + str(math.floor(self.y))

# === Problem 1
class Room(object):
    """
    A Room represents a rectangular region containing clean or dusty
    tiles.

    A room has a width and a height and contains (width * height) tiles. Each tile
    has some fixed amount of dust. The tile is considered clean only when the amount
    of dust on this tile is 0.
    """
    def __init__(self, width, height, dust_amount):
        """
        Initializes a rectangular room with the specified width, height, and
        dust_amount on each tile.

        width: an integer > 0
        height: an integer > 0
        dust_amount: an integer >= 0
        """
        self.width = width
        self.height = height 
        #key represents a tuple (w,h) where w is the x-coord of tile and h is y-coord
        #value represents a value in [0, dust_amount] -- inclusive 
        self.tiles = {}
        for w in range(0, width):
            for h in range(0,height):
                self.tiles[(w,h)] = dust_amount
                

    def get_dust_amount(self, w, h):
        """
        Return the amount of dust on the tile (w, h)

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: a float
        """
        return self.tiles[(w,h)]

    def clean_tile_at_position(self, pos, cleaning_volume):
        """
        Mark the tile under the position pos as cleaned by cleaning_volume amount of dust.

        Assumes that pos represents a valid position inside this room.

        pos: a Position object
        cleaning_volume: a float, the amount of dust to be cleaned in a single time-step.
                  Can be negative which would mean adding dust to the tile.

        Note: The amount of dust on each tile should be NON-NEGATIVE.
              If the cleaning_volume exceeds the amount of dust on the tile, mark it as 0.
        """
        w = math.floor(pos.get_x())
        h = math.floor(pos.get_y())
        old_amt_dust = self.get_dust_amount(w, h)
        
        if old_amt_dust - cleaning_volume > 0:
            self.tiles[(w,h)] = old_amt_dust - cleaning_volume 
        else:
            self.tiles[(w,h)] = 0 


    def is_tile_cleaned(self, w, h):
        """
        Return True if the tile (w, h) has been cleaned.

        Assumes that (w, h) represents a valid tile inside the room.

        w: an integer
        h: an integer

        Returns: True if the tile (w, h) is cleaned, False otherwise

        Note: The tile is considered clean only when the amount of dust on this
              tile is 0.
        """
        return self.tiles[(w,h)] == 0 

    def get_num_cleaned_tiles(self):
        """
        Returns: an integer; the total number of clean tiles in the room
        """
        clean_tiles = 0
        for t in self.tiles.values(): 
            if t == 0:
                clean_tiles += 1 
        return clean_tiles 
                

    def is_position_in_room(self, pos):
        """
        Determines if pos is inside the room.

        pos: a Position object.
        Returns: True if pos is in the room, False otherwise.
        """
        x = pos.get_x()
        y = pos.get_y()

        return (0.0 <= x < self.width) and (0.0 <= y < self.height) 
        

    def get_num_tiles(self):
        """
        Returns: an integer; the total number of tiles in the room
        """
        return self.width*self.height

    def get_random_position(self):
        """
        Returns: a Position object; a random position inside the room
        """
        # w = random.uniform(0.0, float(self.width)-1.0) 
        # h = random.uniform(0.0, float(self.height)-1.0)
        return Position(random.uniform(0.0, float(self.width)), random.uniform(0.0, float(self.height)))

        
    

class Robot(object):
    """
    Represents a robot cleaning a particular room.

    At all times, the robot has a particular position and direction in the room.
    The robot also has a fixed speed and a fixed cleaning_volume.

    Subclasses of Robot should provide movement strategies by implementing
    update_position_and_clean, which simulates a single time-step.
    """
    def __init__(self, room, speed, cleaning_volume):
        """
        Initializes a Robot with the given speed and given cleaning_volume in the
        specified room. The robot initially has a random direction and a random
        position in the room.

        room:  a Room object.
        speed: a positive float.
        cleaning_volume: a positive float; the amount of dust cleaned by the robot
                  in a single time-step.
        pos: a Position object indiciating position of robot in the room 
        """
        self.room = room 
        self.speed = speed
        self.cleaning_volume = cleaning_volume
        self.pos = room.get_random_position()
        self.direction = random.uniform(0.0, 360.0-0.1)

    def get_position(self):
        """
        Returns: a Position object giving the robot's position in the room.
        """
        return self.pos
        

    def get_direction(self):
        """
        Returns: a float d giving the direction of the robot as an angle in
        degrees, 0.0 <= d < 360.0.
        """
        return self.direction 

    def set_position(self, position):
        """
        Set the position of the robot to position.

        position: a Position object.
        """
        self.pos = position 

    def set_direction(self, direction):
        """
        Set the direction of the robot to direction.

        direction: float representing an angle in degrees clockwise from north
        """
        self.direction = direction

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Moves robot to new position and cleans tile according to robot movement
        rules.
        """
        
 
        # DO NOT CHANGE -- implement in subclasses
        raise NotImplementedError

# === Problem 2
class NormalRobot(Robot):
    """
    A NormalRobot is a Robot with the standard movement strategy.

    At each time-step, a NormalRobot attempts to move in its current
    direction; when it would hit a wall, it *instead*
    chooses a new direction randomly.
    """
    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Calculate the next position for the robot.

        If that position is valid, move the robot to that position. Mark the
        tile it is on as having been cleaned by cleaning_volume amount.

        If the new position is invalid, do not move or clean the tile, but
        rotate once to a random new direction.
        """
        new_pos = self.pos.get_new_position(self.direction, self.speed) #does this work??
        if self.room.is_position_in_room(new_pos):
            self.pos = new_pos
            self.room.clean_tile_at_position(self.get_position(), self.cleaning_volume)
            
        else: 
            self.direction = random.uniform(0.0, 360.0-0.1)
            

# === Problem 3
class ClumsyRobot(Robot):
    """
    A ClumsyRobot is a robot that may accidentally drop dust on a tile. A ClumsyRobot will drop some dust 
    on the tile it's on with probability p. 
    The amount of dropped dust should be a random decimal value between 0 and 0.5. 
    Regardless of whether the robot drops dust, it moves to a new position to clean that tile.
    If that new position is not valid, the robot just randomly changes direction.
    """
    p = 0.05

    @staticmethod
    def set_dust_probability(prob):
        """
        Sets the probability of the robot accidentally dropping dust on the tile equal to prob.

        prob: a float (0 <= prob <= 1)
        """
        ClumsyRobot.p = prob

    def does_drop_dust(self):
        """
        Answers the question: Does the robot accidentally drop dust on the tile
        at this timestep?
        The robot drops dust with probability p.

        returns: True if the robot drops dust on its tile, False otherwise.
        """
        return random.random() < ClumsyRobot.p

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        1. Before moving, the robot checks if it should drop dust using does_drop_dust.
            If the robot drops dust, it adds a random decimal amount of dust
            between 0 (inclusive) and 0.5 (exclusive).
            If the robot does not drop dust, it just goes to step 2.
        2. Calculate the robot's next position.
            If the position is valid, the robot cleans that tile.
            If the position is not valid, the robot randomly picks a new direction.
        """
        if self.does_drop_dust(): 
            add_dust = 1
            while add_dust >= 0.5:
                add_dust = random.uniform(0,0.5) #excluedes 0.5??
            self.room.tiles[math.floor(self.pos.get_x()), math.floor(self.pos.get_y())] += add_dust
        
        new_pos = self.pos.get_new_position(self.direction, self.speed)
        if self.room.is_position_in_room(new_pos):
            self.pos = new_pos
            self.room.clean_tile_at_position(self.get_position(), self.cleaning_volume)
             
        else: 
            self.direction = random.uniform(0.0, 360.0-0.1)
                    
            

# === Problem 4
class SensingRobot(Robot):
    """
    A SensingRobot is a robot that can decide which direction to go based on the amount of dust it 
    sees.

    In one timestep, the SensingRobot will look at its surrounding area and find where there is the
    most dust. In the same timestep, it will move towards one of the positions with the most dust.

    It scans the surrounding area by checking each angles between 0 and 360.
    Naturally, many scans will return the same amount of dust, so after the robot completes it's 
    scan, it will randomly pick with uniform probability one of the angles that have the most 
    amount of dust.
    """

    def sense_dust_at_angle(self, angle):
        """
        args:
            angle (int): Angle in degrees of which direction to look

        returns:
            The amount of dust at the position the robot would end up if it moved in the direction
            of angle. Returns -1 if the position is a wall.

        DO NOT MODIFY
        """
        lookahead_pos = self.get_position().get_new_position(angle, self.speed)
        if not self.room.is_position_in_room(lookahead_pos):
            return -1

        tile = (int(lookahead_pos.get_x()), int(lookahead_pos.get_y()))
        return self.room.get_dust_amount(tile[0], tile[1])

    def scan_surrounding_area(self):
        """
        Looks at every 5th integer angle from [0, 360) and returns a dictionary mapping angle to
        dust amount.

        If the scanner sees a wall in one of the scans, it will give a reading of -1.

        DO NOT MODIFY
        """
        dust_amounts = {}
        for angle in range(0, 360, 5):
            dust_amounts[angle] = self.sense_dust_at_angle(angle)

        return dust_amounts

    def update_position_and_clean(self):
        """
        Simulates the passage of a single time-step.

        Within one time step (i.e. one call to update_position_and_clean), the robot should:

        1. Scan the surrounding area (use scan_surrounding_area)
        2. Find the angles with the maximum amount of dust
        3. Pick one of the dirtiest angles at random and move in that direction (random.choice()
           might be useful!)
        4. Clean the tile the robot lands on
        """
        dust_amounts = self.scan_surrounding_area()
        max_dust = max(dust_amounts.values())
        max_angles = []
        for angle in dust_amounts.keys():
            if dust_amounts[angle] == max_dust: 
                max_angles.append(angle)
        angle_chosen = random.choice(max_angles)
        
        new_pos = self.pos.get_new_position(angle_chosen, self.speed)
        if self.room.is_position_in_room(new_pos):
            self.pos = new_pos
            self.room.clean_tile_at_position(self.get_position(), self.cleaning_volume)
             
        else: 
            self.direction = random.uniform(0.0, 360.0-0.1)
        
                

# === Problem 5
def run_simulation(num_robots, speed, cleaning_volume, width, height, dust_amount, min_coverage, num_trials,
                  robot_type):
    """
    Runs num_trials trials of the simulation and returns the mean number of
    time-steps needed to clean the fraction min_coverage of the room.

    The simulation is run with num_robots robots of type robot_type, each
    with the input speed and cleaning_volume in a room of dimensions width x height
    with the dust dust_amount on each tile. Each trial is run in its own Room
    with its own robots.

    num_robots: an int (num_robots > 0)
    speed: a float (speed > 0)
    cleaning_volume: a float (cleaning_volume > 0)
    width: an int (width > 0)
    height: an int (height > 0)
    dust_amount: an int
    min_coverage: a float (0 <= min_coverage <= 1.0)
    num_trials: an int (num_trials > 0)
    robot_type: class of robot to be instantiated (e.g. NormalRobot or
                ClumsyRobot)
    """
    #Simulates the robot(s) cleaning the room up to a specified fraction of the room for various trials; and
    #Returns the mean number of time-steps needed to clean the room.
    
    total_time_steps = []
    for n in range(num_trials):
        
        # Each trial should begin with a new room and num_robots number of robots.
        # Each robot in the trial should start at a random position in that room.
        # Each room should start with a uniform amount of dust on each tile, given by dust_amount.
        robots = []
        room = Room(width, height, dust_amount)
        #print(room)
        for r in range(num_robots):
            robots.append(robot_type(room, speed, cleaning_volume))
        time_step = 0 
        while room.get_num_cleaned_tiles() < min_coverage*room.get_num_tiles(): 
            #print(room.get_num_cleaned_tiles())
            for r in robots:
                r.update_position_and_clean() 
            time_step += 1
        total_time_steps.append(time_step)
    return sum(total_time_steps)/num_trials
        
                
            

    
    # A trial ends when a specified fraction of the room tiles have been fully cleaned 
    # (i.e., the amount of dust on those tiles is 0). The simulation terminates once the specified 
    # number of trials has been run. The average number of time steps for the simulation is the sum 
    # of the time steps across all the trials over the number of trials.
    

#print('avg time steps: ' + str(run_simulation(1, 1.0, 1, 2, 1, 3, 1.0, 10, NormalRobot)))
#print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 5, 5, 3, 1.0, 50, NormalRobot)))
#print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.8, 50, NormalRobot)))
#print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 10, 10, 3, 0.9, 50, NormalRobot)))
#print ('avg time steps: ' + str(run_simulation(1, 1.0, 1, 20, 20, 3, 0.5, 50, NormalRobot)))
#print ('avg time steps: ' + str(run_simulation(3, 1.0, 1, 20, 20, 3, 0.5, 50, NormalRobot)))

# === Problem 6
#
# ANSWER THE FOLLOWING QUESTIONS:
#
# 1)How does the performance of the three robot types compare when cleaning 80%
#       of a 20x20 room?
#   Sensing Robot shows the fastest performance taking an average of less time steps to clean the room,
#   followed by Normal Robot, and Clumsy having the slowest performance and an average of most time steps to clean the room. 
#   All robots show a pattern of decreasing the numbner of time steps with an increase number of robots, with clumsy robot 
#   showing the greatest decrease number of time steps. 

# 2) How does the performance of the three robot types compare when two of each
#       robot cleans 80% of rooms with dimensions
#       10x30, 20x15, 25x12, and 50x6?
#   
#   Shows a similar trend to the previous plot where SensingRobot hasthe fastest performance,
#   and ClumsyRobot is the slowest robot. However, here see a slight decrease in efficiency for NormalRobot
#   and ClumsyRobot types as the ratio of the room (w:h) increases. Sensing Robot has on average the highest 
#   increase in time steps as the ratio increases. 

def show_plot_compare_strategies(title, x_label, y_label):
    """
    Produces a plot comparing the three robot strategies in a 20x20 room with 80%
    minimum coverage.
    """
    num_robot_range = range(1, 3)
    times1 = []
    times2 = []
    times3 = []
    for num_robots in num_robot_range:
        print ("Plotting", num_robots, "robots...")
        times1.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, NormalRobot))
        times2.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, ClumsyRobot))
        times3.append(run_simulation(num_robots, 1.0, 1, 20, 20, 3, 0.8, 20, SensingRobot))
    pylab.plot(num_robot_range, times1)
    pylab.plot(num_robot_range, times2)
    pylab.plot(num_robot_range, times3)
    pylab.title(title)
    pylab.legend(('NormalRobot', 'ClumsyRobot', 'SensingRobot'))
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

def show_plot_room_shape(title, x_label, y_label):
    """
    Produces a plot showing dependence of cleaning time on room shape.
    """
    aspect_ratios = []
    times1 = []
    times2 = []
    times3 = []
    for width in [10, 20, 25, 50]:
        height = int(300/width)
        print ("Plotting cleaning time for a room of width:", width, "by height:", height)
        aspect_ratios.append(float(width) / height)
        times1.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, NormalRobot))
        times2.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, ClumsyRobot))
        times3.append(run_simulation(2, 1.0, 1, width, height, 3, 0.8, 200, SensingRobot))
    pylab.plot(aspect_ratios, times1, 'o-')
    pylab.plot(aspect_ratios, times2, 'o-')
    pylab.plot(aspect_ratios, times3, 'o-')
    pylab.title(title)
    pylab.legend(('NormalRobot', 'ClumsyRobot', 'SensingRobot'), fancybox=True, framealpha=0.5)
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    pylab.show()

if __name__ == "__main__":
    # Test code should go HERE so that test_ps3.py doesn't have to run it!
    # All code in this block will be run ONLY if you run ps3.py directly.

    # Uncomment this line to see your implementation of NormalRobot in action!
    # test_robot_movement(NormalRobot, Room)

    # Uncomment this line to see your implementation of ClumsyRobot in action!
    #test_robot_movement(ClumsyRobot, Room)
    
    # Uncomment this line to see your implementation of SensingRobot in action!
    #test_robot_movement(SensingRobot, Room)

    #show_plot_compare_strategies('Time to clean 80% of a 20x20 room, for various numbers of robots','Number of robots','Time (steps)')
    #show_plot_room_shape('Time to clean 80% of a 300-tile room for various room shapes','Aspect Ratio', 'Time (steps)')

    pass
