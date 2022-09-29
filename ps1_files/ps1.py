################################################################################
# 6.0002 Spring 2022
# Problem Set 1
# Name: Jessica Jimenez
# Collaborators: None
# Time: 9 hrs 

from state import State

##########################################################################################################
## Problem 1
##########################################################################################################

def load_election(filename):
    """
    Reads the contents of a file, with data given in the following tab-separated format:
    State[tab]Democrat_votes[tab]Republican_votes[tab]EC_votes

    Please ignore the first line of the file, which are the column headers, and remember that
    the special character for tab is '\t'

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """
    result = []
    f = open(filename)
    f.readline()
    #looped through each line in file (each line is an instance of State)
    for line in f: 
        row = line.split("\t")
        result.append(State(row[0],row[1], row[2], row[3].strip('/t')))
        
    return result


##########################################################################################################
## Problem 2: Helper functions
##########################################################################################################

def election_winner(election):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election - a list of State instances

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'rep') if Democrats won, else ('rep', 'dem')
    """
    dem_votes = 0
    rep_votes = 0
    #loop through each state and find total number of ec for each party
    for s in election:
        #give votes to the winner of election
        if s.get_winner() == 'dem':
            dem_votes += s.get_ecvotes()
        else:
            rep_votes += s.get_ecvotes()
    if dem_votes > rep_votes:
        return ('dem', 'rep')
    else:
        return ('rep', 'dem')



def winner_states(election):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).

    Parameters:
    election - a list of State instances

    Returns:
    A list of State instances won by the winning candidate
    """
    
    winner = election_winner(election)[0]
    states_win = []
    
    #loop through each state and returns list of states won by candidate who won. 
    for s in election:
        if s.get_winner() == winner:
            states_win.append(s)
    return states_win
            


def ec_votes_to_flip(election, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election - a list of State instances
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """
    loser = election_winner(election)[1]
    loser_votes = 0
    #loop through states and find total number votes received by loser
    for s in election:
        if s.get_winner() == loser:
            loser_votes += s.get_ecvotes()
    #find diff of votes needed to win with 1/2EC + 1 votes 
    return int((total/2+1) - loser_votes) 
    
     


##########################################################################################################
## Problem 3: Brute Force approach
##########################################################################################################

def combinations(L):
    """
    Helper function to generate powerset of all possible combinations
    of items in input list L. E.g., if
    L is [1, 2] it will return a list with elements
    [], [1], [2], and [1,2].

    DO NOT MODIFY THIS.

    Parameters:
    L - list of items

    Returns:
    a list of lists that contains all possible
    combinations of the elements of L
    """

    def get_binary_representation(n, num_digits):
        """
        Inner function to get a binary representation of items to add to a subset,
        which combinations() uses to construct and append another item to the powerset.

        DO NOT MODIFY THIS.

        Parameters:
        n and num_digits are non-negative ints

        Returns:
            a num_digits str that is a binary representation of n
        """
        result = ''
        while n > 0:
            result = str(n%2) + result
            n = n//2
        if len(result) > num_digits:
            raise ValueError('not enough digits')
        for i in range(num_digits - len(result)):
            result = '0' + result
        return result

    powerset = []
    for i in range(0, 2**len(L)):
        binStr = get_binary_representation(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def brute_force_swing_states(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states, these are our swing states. Iterate over
    all possible move combinations using the helper function combinations(L).
    Return the move combination that minimises the number of voters moved. If
    there exists more than one combination that minimises this, return any one of them.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances such that the election outcome would change if additional
      voters relocated to those states, as well as the number of voters required for that relocation.
    * A tuple containing the empty list followed by zero, if no possible swing states.
    """
    best_combo = ([], None)
    #combos of winning states that can be flipped
    possible_flipped = combinations(winner_states)
    for combo in possible_flipped: 
        if combo: #not empty
            voters_moved = 0
            ecvotes_flipped = 0
            #loop through ea state in combo
            for s in combo:
                #in each combo find # of voters moved and ecvotes flipped 
                voters_moved += s.get_margin() + 1
                ecvotes_flipped += s.get_ecvotes()
             #if the #of ec votes is at min the number needed   
            if ecvotes_flipped >= ec_votes_needed:   
                #check if this combo is the best combo by checking # of voters moved
                if best_combo[1] == None:
                    best_combo = combo, voters_moved
                elif (voters_moved < best_combo[1]):
                    best_combo = combo, voters_moved
    #after looping through all combinations return best one            
    return best_combo 

            
        


##########################################################################################################
## Problem 4: Dynamic Programming
## In this section we will define two functions, max_voters_moved and min_voters_moved, that
## together will provide a dynamic programming approach to find swing states. This problem
## is analagous to the complementary knapsack problem, you might find Lecture 1 of 6.0002 useful
## for this section of the pset.
##########################################################################################################


def max_voters_moved(winner_states, max_ec_votes, memo = None):
    """
    Finds the largest number of voters needed to relocate to get at most max_ec_votes
    for the election loser.

    Analogy to the knapsack problem:
        Given a list of states each with a weight(ec_votes) and value(margin+1),
        determine the states to include in a collection so the total weight(ec_votes)
        is less than or equal to the given limit(max_ec_votes) and the total value(voters displaced)
        is as large as possible.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    max_ec_votes - int, the maximum number of EC votes

    Returns:
    * A tuple containing the list of State instances such that the maximum number of voters need to
      be relocated to these states in order to get at most max_ec_votes, and the number of voters
      required required for such a relocation.
    * A tuple containing the empty list followed by zero, if every state has a # EC votes greater
      than max_ec_votes.
    """
    #keep a memo that tracks length of # of winner states and maximum ec votes 
    if memo == None: 
        memo = {}
    if winner_states == [] or max_ec_votes == 0: 
        result = ([], 0)
    elif (len(winner_states), max_ec_votes) in memo: 
        return memo[(len(winner_states), max_ec_votes)]
    elif winner_states[0].get_ecvotes() > max_ec_votes:
        return max_voters_moved(winner_states[1:], max_ec_votes, memo)
    else: 
        #explore tree if state is flipped
        curr_state = winner_states[0] 
        flip_states, flip_voters =  max_voters_moved(winner_states[1:], max_ec_votes-curr_state.get_ecvotes(), memo)
        flip_voters += curr_state.get_margin()+1
        
        #explore tree if state is not flipped
        unflip_states, unflip_voters = max_voters_moved(winner_states[1:], max_ec_votes, memo)
        
        #choose better choice to be result 
        if flip_voters > unflip_voters: 
            result = flip_states+[curr_state], flip_voters
        else:
            result =  unflip_states, unflip_voters
    #keep track of all results
    memo[(len(winner_states), max_ec_votes)] = result 
    
    return result 

    

def min_voters_moved(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated.
    Only return states that were originally won by the winner (lost by the loser)
    of the election.

    Hint: This problem is simply the complement of max_voters_moved. You should call
    max_voters_moved with max_ec_votes set to (#ec votes won by original winner - ec_votes_needed)

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    * A tuple containing the list of State instances (which we can call swing states) such that the
      minimum number of voters need to be relocated to these states in order to get at least
      ec_votes_needed, and the number of voters required for such a relocation.
    * * A tuple containing the empty list followed by zero, if no possible swing states.
    """
    original_winner_ec = 0
    
    #find # of ec votes winne originally had 
    for s in winner_states:
        original_winner_ec += s.get_ecvotes()
    #find list of states where voters will not move out of 
    non_swing_states, non_voters_relocated = max_voters_moved(winner_states, original_winner_ec - ec_votes_needed)
    swing_states = []
    #loop through ea winner state and make a list of swing states (states not in non_swing)
    for s in winner_states:
        if s not in non_swing_states:
            swing_states.append(s)
    #if a swing states is not empty find minimum voters needed to move 
    if swing_states:
        min_voters_move = 0
        for s in swing_states:
            min_voters_move += s.get_margin()+1
        
        return (swing_states, min_voters_move)
    else:
        return ([], 0)
   



##########################################################################################################
## Problem 5
##########################################################################################################


def relocate_voters(election, swing_states, ideal_states = ['AL', 'AZ', 'CA', 'TX']):
    """
    Finds a way to shuffle voters in order to flip an election outcome. Moves voters
    from states that were won by the losing candidate (states not in winner_states), to
    each of the states in swing_states. To win a swing state, you must move (margin + 1)
    new voters into that state. Any state that voters are moved from should still be won
    by the loser even after voters are moved. Also finds the number of EC votes gained by
    this rearrangement, as well as the minimum number of voters that need to be moved.
    Note: You cannot move voters out of Alabama, Arizona, California, or Texas.

    Parameters:
    election - a list of State instances representing the election
    swing_states - a list of State instances where people need to move to flip the election outcome
                   (result of min_voters_moved or brute_force_swing_states)
    ideal_states - a list of Strings holding the names of states where residents cannot be moved from
                   (default states are AL, AZ, CA, TX)

    Return:
    * A tuple that has 3 elements in the following order:
        - an int, the total number of voters moved
        - an int, the total number of EC votes gained by moving the voters
        - a dictionary with the following (key, value) mapping:
            - Key: a 2 element tuple of str, (from_state, to_state), the 2 letter State names
            - Value: int, number of people that are being moved
    * None, if it is not possible to sway the election
    """
    #states that voters can move out from to change result 
    losing_candidate_states = [s for s in election if (s.get_name() not in ideal_states and s not in swing_states)]
    #states that voters can move into 
    swing_states_copy = [s for s in swing_states if (s not in ideal_states)]
    winner = swing_states_copy[0].get_winner()

    movement = {}
    total_voters_moved = 0
    total_ec_gained = 0
    
    #loop through ea. state voters can move out of 
    for ls in losing_candidate_states: 
        #loop through ea. state voters can move into 
        for ss in swing_states_copy:
            #if swing state still has orig winner and voters left to relocate 
            if ss.get_winner() == winner and ss.get_margin() > 1:
                ss_voters_to_swing = ss.get_margin()+1
                #if amount of voters want to relocate is less than max votes the losing state can reloc
                if ss_voters_to_swing <= ls.get_margin()-1:
                    movement[(ls.get_name(), ss.get_name())] = ss_voters_to_swing
                    ss.add_losing_candidate_voters(ss_voters_to_swing)
                    ls.subtract_winning_candidate_voters(ss_voters_to_swing)
                    total_voters_moved += ss_voters_to_swing
                    total_ec_gained += ss.get_ecvotes()
                    #if no more votes left to swing break and move on to next loser's state
                    if ss_voters_to_swing == ls.get_margin()-1:
                        break
                #if amount of voters needed to reloc is more than max votes the losing state can reloc
                elif ls.get_margin() > 1:
                    #move max amount of voters from loser state to swing and move onto next loser's state
                    ls_to_move = ls.get_margin()-1
                    movement[(ls.get_name(), ss.get_name())] = ls_to_move
                    ss.add_losing_candidate_voters(ls_to_move)
                    ls.subtract_winning_candidate_voters(ls_to_move)
                    total_voters_moved += ls_to_move
                    break 
    if total_voters_moved != 0:   
        return (total_voters_moved, total_ec_gained, movement) 
    else:
        return None
                

            
                
                


if __name__ == "__main__":
    # Uncomment the following lines to test each of the problems
    
    # # tests Problem 1
    year = 2012
    election = load_election(f"{year}_results.txt")
    print(len(election))
    print(election[0])
    
    # # tests Problem 2
    winner, loser = election_winner(election)
    won_states = winner_states(election)
    names_won_states = [state.get_name() for state in won_states]
    reqd_ec_votes = ec_votes_to_flip(election)
    print("Winner:", winner, "\nLoser:", loser)
    print("States won by the winner: ", names_won_states)
    print("EC votes needed:",reqd_ec_votes, "\n")
    
    # # tests Problem 3
    brute_election = load_election("60002_results.txt")
    brute_won_states = winner_states(brute_election)
    brute_ec_votes_to_flip = ec_votes_to_flip(brute_election, total=14)
    brute_swing, voters_brute = brute_force_swing_states(brute_won_states, brute_ec_votes_to_flip)
    names_brute_swing = [state.get_name() for state in brute_swing]
    ecvotes_brute = sum([state.get_ecvotes() for state in brute_swing])
    print("Brute force swing states results:", names_brute_swing)
    print("Brute force voters displaced:", voters_brute, "for a total of", ecvotes_brute, "Electoral College votes.\n")
    
    # # tests Problem 4a: max_voters_moved
    print("max_voters_moved")
    total_lost = sum(state.get_ecvotes() for state in won_states)
    non_swing_states, max_voters_displaced = max_voters_moved(won_states, total_lost-reqd_ec_votes)
    non_swing_states_names = [state.get_name() for state in non_swing_states]
    max_ec_votes = sum([state.get_ecvotes() for state in non_swing_states])
    print("States with the largest margins (non-swing states):", non_swing_states_names)
    print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")
    
    # # tests Problem 4b: min_voters_moved
    print("min_voters_moved")
    swing_states, min_voters_displaced = min_voters_moved(won_states, reqd_ec_votes)
    swing_state_names = [state.get_name() for state in swing_states]
    swing_ec_votes = sum([state.get_ecvotes() for state in swing_states])
    print("Complementary knapsack swing states results:", swing_state_names)
    print("Min voters displaced:", min_voters_displaced, "for a total of", swing_ec_votes, "Electoral College votes. \n")
    
    # tests Problem 5: relocate_voters
    print("relocate_voters")
    flipped_election = relocate_voters(election, swing_states)
    print("Flip election mapping:", flipped_election)