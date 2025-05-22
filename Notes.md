# Fixed
1. Removed win Probability from match_result
2. phases error fixed, 4 phases throughout

# To fix
1. Probability calculation
2. Database integration

# To Evaluate
1. Probability calculation/predictions and Cross entropy loss

1. Batting order selection function that creates a dataframe with 11 positions each contain a list of dictionaries of Batters ranked by thei rperformance score in that position. Assume that by_position is a key for every batter in the batting stats json. Use a combination of these stats and by_phase stats to calculate a performance score for each batter for each position. So this list of dictionaries will contain
2. Playing XI selection function that picks the best XI players from the squad. This function take multiple arguments like ground, opponent e.t.c. but all will be optional and this playing X! should be a variable thats initialized upon creation of an instance of the object
Logic for playing XI selection: