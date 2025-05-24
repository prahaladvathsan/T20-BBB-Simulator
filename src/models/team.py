"""T20 Cricket Match Simulator: Team Model
This module defines the Team class with objective team selection and strategic decision-making capabilities.
Uses optimization algorithms to select optimal XI and create batting and bowling strategies.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from scipy.optimize import linear_sum_assignment  # For Hungarian algorithm


class Team:
    """
    Team class with objective team selection and advanced strategy methods for cricket match simulation.
    """
    
    # CONSTANTS
    # Phase definitions
    POWERPLAY = 1      # Overs 1-6
    EARLY_MIDDLE = 2   # Overs 7-12  
    LATE_MIDDLE = 3    # Overs 13-16
    DEATH = 4          # Overs 17-20
    
    PHASE_OVERS = {
        POWERPLAY: (1, 6),
        EARLY_MIDDLE: (7, 12),
        LATE_MIDDLE: (13, 16),
        DEATH: (17, 20)
    }

    
    # Position importance weights for batting
    BATTING_POSITION_WEIGHTS = {
        1: 0.170919,
        2: 0.196982,
        3: 0.163785,
        4: 0.142512,
        5: 0.121843,
        6: 0.085424,
        7: 0.062353,
        8: 0.033629,
        9: 0.015203,
        10: 0.005149,
        11: 0.002201
    }

    

    
    def __init__(self, team_id: str, team_data: Dict, player_objects: Dict):
        """Initialize team with players and strategies."""
        # Basic team information
        self.id = team_id
        self.name = team_data.get('name', f"Team {team_id}")
        self.player_ids = team_data.get('players', [])
        self.players = {player_id: player_objects[player_id] for player_id in self.player_ids if player_id in player_objects}
        
        # Initialize squad variable to store all player details
        self.squad = {}
        for player_id, player in self.players.items():
            self.squad[player_id] = {
                'id': player_id,
                'name': player.name,
                'main_role': player.main_role,
                'is_wicketkeeper': player.main_role == 'WK-Batter',
                'can_bowl': hasattr(player, 'bowling_stats'),
                'batting_roles': [],
                'bowling_roles': [],
                'stats': {
                    'batting': player.batting_stats if hasattr(player, 'batting_stats') else {},
                    'bowling': player.bowling_stats if hasattr(player, 'bowling_stats') else {}
                }
            
            }

        # Initialize role lists
        self.batsmen = []
        self.bowlers = []
        self.all_rounders = []
        self.wicket_keepers = []
        
        # Initialize role assignments
        self._assign_main_roles()
        self._assign_batting_roles()
        self._assign_bowling_roles()
        
        # Create detailed performance rankings
        self.batting_position_rankings = self._create_batting_position_rankings()
        self.bowling_over_rankings = self._create_bowling_over_rankings()
        
        # Select optimal XI team
        self.optimal_xi, self.optimal_batting_order, self.optimal_bowling_allocation = self.pick_optimal_xi()
    
    def _assign_main_roles(self):
        """Assign main roles to players based on their main_role attribute."""
        for player_id, player in self.players.items():
            if player.main_role == 'Batter':
                self.batsmen.append(player_id)
            elif player.main_role == 'Bowler':
                self.bowlers.append(player_id)
            elif player.main_role == 'All-Rounder':
                self.all_rounders.append(player_id)
            elif player.main_role == 'WK-Batter':
                self.wicket_keepers.append(player_id)
                if player_id not in self.batsmen:
                    self.batsmen.append(player_id)
    
    def _assign_batting_roles(self):
        """Assign batting roles based on effective average and strike rate."""
        for player_id, player in self.players.items():
            if hasattr(player, 'batting_stats'):
                effective_avg = player.batting_stats.get('effective_average', 0)
                effective_sr = player.batting_stats.get('effective_strike_rate', 0)
                if effective_avg and effective_sr is not None:
                    if effective_avg > 120:
                        self.squad[player_id]['batting_roles'].append('Anchor')
                    if effective_sr > 120:
                        self.squad[player_id]['batting_roles'].append('Pinch Hitter')
    
    def _assign_bowling_roles(self):
        """Assign bowling roles based on effective strike rate and economy."""
        for player_id, player in self.players.items():
            if hasattr(player, 'bowling_stats'):
                effective_sr = player.bowling_stats.get('effective_strike_rate', 999)
                effective_econ = player.bowling_stats.get('effective_economy', 999)
                
                if effective_sr < 80:
                    self.squad[player_id]['bowling_roles'].append('Strike Bowler')
                if effective_econ < 8:
                    self.squad[player_id]['bowling_roles'].append('Defensive Bowler')
    
    def _calculate_batting_position_score(self, player: Any, position: int) -> float:
        """
        Calculate a player's performance score for a specific batting position.
        
        Args:
            player: Player object
            position: Batting position (1-11)
            
        Returns:
            float: Performance score for the position
        """
        if not hasattr(player, 'batting_stats'):
            return 0.0
            
        stats = player.batting_stats
        position_stats = stats.get('by_bat_pos', {}).get(str(position), {})
        
        # Read phase weights from CSV
        phase_weights_df = pd.read_csv('data/processed/phase_weights.csv')
        phase_weights = {
            '1': phase_weights_df.iloc[position-1, 1] / 100,  # Powerplay
            '2': phase_weights_df.iloc[position-1, 2] / 100,  # Early middle
            '3': phase_weights_df.iloc[position-1, 3] / 100,  # Late middle
            '4': phase_weights_df.iloc[position-1, 4] / 100   # Death overs
        }
            
        # Calculate weighted phase stats
        weighted_phase_average = 0
        weighted_phase_sr = 0
        
        for phase in ['1', '2', '3', '4']:
            phase_stats = stats.get('by_phase', {}).get(phase, {})
            phase_runs = phase_stats.get('runs', 0)
            phase_balls = phase_stats.get('balls', 0)
            phase_dismissals = phase_stats.get('dismissals', 1)  # Avoid division by zero
            
            phase_average = phase_runs / phase_dismissals if phase_dismissals > 0 else 0
            phase_sr = (phase_runs / phase_balls * 100) if phase_balls > 0 else 0
            
            weighted_phase_average += phase_average * phase_weights[phase]
            weighted_phase_sr += phase_sr * phase_weights[phase]
        
        # Get position-specific stats
        pos_average = position_stats.get('average', weighted_phase_average * 0.9)
        pos_sr = position_stats.get('strike_rate', weighted_phase_sr * 0.9) 
        
        # Calculate weighted score with fallbacks
        position_score = (
            pos_average * 0.2 +           # 30% weight to position average
            pos_sr * 0.4 +                # 20% weight to position SR
            weighted_phase_average * 0.1 + # 20% weight to phase average
            weighted_phase_sr * 0.2 +      # 20% weight to phase SR
            stats.get('boundary_percentage', 0) * 0.1  # 10% weight to boundary hitting
        )
        
        # Ensure the score is valid and non-negative
        return max(0.0, float(position_score))
    
    def _create_batting_position_rankings(self) -> pd.DataFrame:
        """
        Create a DataFrame with rankings for each batting position.
        
        Returns:
            DataFrame with 11 positions, each containing sorted list of player performances
        """
        # Initialize DataFrame with 11 positions
        positions = range(1, 12)
        rankings = pd.DataFrame(index=positions, columns=['rankings'])
        
        # For each position, calculate and sort player performances
        for position in positions:
            position_rankings = []
            
            # Calculate scores for all players
            for player_id, player in self.players.items():
                score = self._calculate_batting_position_score(player, position)
                if score > 0:  # Only include players with valid scores
                    position_rankings.append({
                        'player_id': player_id,
                        'name': player.name,
                        'main_role': player.main_role,
                        'score': score,
                        'position_stats': player.batting_stats.get('by_bat_pos', {}).get(str(position), {}),
                        'batting_roles': self.squad[player_id]['batting_roles'],
                        'is_wicketkeeper': self.squad[player_id]['is_wicketkeeper']
                    })
            
            # Sort by score in descending order
            position_rankings.sort(key=lambda x: x['score'], reverse=True)
            rankings.at[position, 'rankings'] = position_rankings
        
        return rankings
    
    def _calculate_bowling_over_score(self, player: Any, over: int) -> float:
        """
        Calculate a player's performance score for a specific over.
        
        Args:
            player: Player object
            over: Over number (1-20)
            
        Returns:
            float: Performance score for the over
        """
        if not hasattr(player, 'bowling_stats'):
            return 0.0
            
        stats = player.bowling_stats
        over_stats = stats.get('by_over', {}).get(str(over), {})
        
        # Get phase-specific stats based on over
        if over <= 6:
            phase = '1'  # Powerplay
        elif over <= 12:
            phase = '2'  # Early middle
        elif over <= 16:
            phase = '3'  # Late middle
        else:
            phase = '4'  # Death overs
            
        phase_stats = stats.get('by_phase', {}).get(phase, {})
        
        # Calculate weighted score (higher is better for bowling too)
        over_score = (
            (10 - over_stats.get('economy', 10)) * 0.3 +      # 30% weight to over economy
            (50 - over_stats.get('average', 50)) * 0.2 +      # 20% weight to over average
            (10 - phase_stats.get('economy', 10)) * 0.2 +     # 20% weight to phase economy
            (50 - phase_stats.get('average', 50)) * 0.2 +     # 20% weight to phase average
            over_stats.get('dot_percentage', 0) * 0.1         # 10% weight to dot balls
        )
        
        return max(0, over_score)  # Ensure non-negative score
    
    def _create_bowling_over_rankings(self) -> pd.DataFrame:
        """
        Create a DataFrame with rankings for each over.
        
        Returns:
            DataFrame with 20 overs, each containing sorted list of bowler performances
        """
        # Initialize DataFrame with 20 overs
        overs = range(1, 21)
        rankings = pd.DataFrame(index=overs, columns=['rankings'])
        
        # For each over, calculate and sort bowler performances
        for over in overs:
            over_rankings = []
            
            # Calculate scores for all players
            for player_id, player in self.players.items():
                score = self._calculate_bowling_over_score(player, over)
                over_rankings.append({
                    'player_id': player_id,
                    'name': player.name,
                    'main_role': player.main_role,
                    'score': score,
                    'over_stats': player.bowling_stats.get('by_over', {}).get(str(over), {}),
                    'bowling_roles': self.squad[player_id]['bowling_roles']
                })

            # Sort by score in descending order
            over_rankings.sort(key=lambda x: x['score'], reverse=True)
            rankings.at[over, 'rankings'] = over_rankings
        
        return rankings
    
    def pick_optimal_xi_v1(self) -> Tuple[List[str], List[str], Dict[int, str]]:
        """
        Select the optimal playing XI, batting order, and bowling allocation.
        
        Returns:
            Tuple containing:
            - List of player IDs in the optimal XI
            - List of player IDs in optimal batting order
            - Dict mapping overs to bowler IDs for optimal bowling allocation
        """
        # 1. Calculate theoretical optimal batting and bowling
        # Get theoretical max batting performance
        theoretical_batting_order, max_batting_score = self._optimize_batting_order_unlimited()
        print("Theoretical batting order:")
        print([self.players[player_id].name for player_id in theoretical_batting_order])
        print(max_batting_score)
        
        # Get theoretical max bowling performance and allocation
        theoretical_bowling_allocation, max_bowling_score = self._optimize_bowling_allocation_unlimited()
        print("Theoretical bowling allocation:")
        print(theoretical_bowling_allocation)
        print(max_bowling_score)
        # 2. Start by identifying constraints
        # Get all players who can bowl
        bowlers = set(self.bowlers + self.all_rounders)
        
        # Get all wicketkeepers
        wicketkeepers = set(self.wicket_keepers)
        
        # 3. Search for best combination satisfying constraints
        # Use simulated annealing for optimization
        best_xi, best_batting_order, best_bowling_allocation, best_score = self._optimize_team_selection(
            bowlers, wicketkeepers, max_batting_score, max_bowling_score, 
            theoretical_batting_order, theoretical_bowling_allocation
        )
        
        return best_xi, best_batting_order, best_bowling_allocation
    
    def _optimize_batting_order_unlimited(self) -> Tuple[List[str], float]:
        """
        Optimize batting order using the Hungarian algorithm without player limitations.
        
        Returns:
            Tuple containing:
            - Optimal batting order (list of player IDs)
            - Total batting score
        """
        # Get all available batsmen
        all_batsmen = set()
        for position in range(1, 12):
            for player in self.batting_position_rankings.at[position, 'rankings']:
                all_batsmen.add(player['player_id'])
        
        # Create cost matrix for Hungarian algorithm
        cost_matrix = []
        batsmen_list = list(all_batsmen)
        
        for batsman_id in batsmen_list:
            batsman_costs = []
            for position in range(1, 12):
                # Find this player's entry for this position
                player_entry = next(
                    (p for p in self.batting_position_rankings.at[position, 'rankings'] 
                     if p['player_id'] == batsman_id), 
                    None
                )
                
                if player_entry:
                    # Negative because we're maximizing but Hungarian minimizes
                    weighted_score = -player_entry['score'] * self.BATTING_POSITION_WEIGHTS[position]
                    batsman_costs.append(weighted_score)
                else:
                    # Very high cost if player has no score for this position
                    batsman_costs.append(1000000)
            
            cost_matrix.append(batsman_costs)
        
        # Run Hungarian algorithm
        if not cost_matrix:
            return [], 0.0
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create batting order
        batting_order = [None] * 11
        total_score = 0.0
        
        for i, position_idx in enumerate(col_ind):
            if i < len(batsmen_list):
                batsman_id = batsmen_list[row_ind[i]]
                position = position_idx + 1
                
                # Find player's actual score for this position
                player_entry = next(
                    (p for p in self.batting_position_rankings.at[position, 'rankings'] 
                     if p['player_id'] == batsman_id), 
                    None
                )
                
                if player_entry:
                    batting_order[position_idx] = batsman_id
                    total_score += player_entry['score']

        
        return batting_order, total_score
    
    def _optimize_bowling_allocation_unlimited(self) -> Tuple[Dict[int, str], float]:
        """
        Optimize bowling allocation without player limitations.
        Uses integer linear programming principles implemented with a greedy algorithm.
        
        Returns:
            Tuple containing:
            - Bowling allocation (dict: over -> bowler_id)
            - Total bowling score
        """
        # Initialize allocation and tracking variables
        allocation = {}
        overs_bowled = {}
        last_bowler = None
        total_score = 0.0
        
        # Loop through overs
        for over in range(1, 21):
            best_score = -1
            best_bowler = None
            
            # Check all bowlers for this over
            for player in self.bowling_over_rankings.at[over, 'rankings']:
                bowler_id = player['player_id']
                # Check constraints
                if (overs_bowled.get(bowler_id, 0) < 4 and 
                    last_bowler != bowler_id):
                    
                    if player['score'] > best_score:
                        best_score = player['score']
                        best_bowler = bowler_id
            
            # Assign best bowler to this over
            if best_bowler:
                allocation[over] = best_bowler
                overs_bowled[best_bowler] = overs_bowled.get(best_bowler, 0) + 1
                last_bowler = best_bowler
                print(f"Over {over} assigned to {self.players[best_bowler].name} with score {best_score}")
                total_score += best_score
            else:
                # If no valid bowler found, allow the same bowler
                # This is a fallback that shouldn't normally happen
                for player in self.bowling_over_rankings.at[over, 'rankings']:
                    bowler_id = player['player_id']
                    
                    if overs_bowled.get(bowler_id, 0) < 4:
                        allocation[over] = bowler_id
                        overs_bowled[bowler_id] = overs_bowled.get(bowler_id, 0) + 1
                        last_bowler = bowler_id
                        total_score += player['score']
                        break
        
        print(f"Total bowling score: {total_score}")
        return allocation, total_score
    
    def _optimize_team_selection_v1(self, bowlers, wicketkeepers, max_batting_score, max_bowling_score,
                               theoretical_batting_order, theoretical_bowling_allocation):
        """
        Use simulated annealing to find the optimal team selection.
        
        Args:
            bowlers: Set of player IDs who can bowl
            wicketkeepers: Set of wicketkeeper IDs
            max_batting_score: Theoretical maximum batting score
            max_bowling_score: Theoretical maximum bowling score
            theoretical_batting_order: Unconstrained optimal batting order
            theoretical_bowling_allocation: Unconstrained optimal bowling allocation
            
        Returns:
            Tuple containing:
            - Optimal XI (list of player IDs)
            - Optimal batting order (list of player IDs)
            - Optimal bowling allocation (dict: over -> bowler_id)
            - Combined performance score
        """
        # Define parameters for simulated annealing
        initial_temperature = 100.0
        final_temperature = 0.1
        cooling_rate = 0.95
        iterations_per_temp = 50
        
        # Start with a valid random team
        current_team = self._generate_valid_team(bowlers, wicketkeepers)

        # Print team names and not IDs
        print([self.players[player_id].name for player_id in current_team])
        
        # Evaluate current team
        current_batting_order, current_batting_score = self._optimize_batting_order(current_team)
        current_bowling_allocation, current_bowling_score = self._optimize_bowling_allocation(current_team)

        print([self.players[player_id].name for player_id in current_batting_order])
        
        # Calculate combined normalized score
        current_score = (current_batting_score / max_batting_score + 
                        current_bowling_score / max_bowling_score)
        
        # Initialize best solution
        best_team = current_team.copy()
        best_batting_order = current_batting_order.copy()
        best_bowling_allocation = current_bowling_allocation.copy()
        best_score = current_score
        print(best_score)
        
        # Main simulated annealing loop
        temperature = initial_temperature
        while temperature > final_temperature:
            for _ in range(iterations_per_temp):
                # Generate a neighbor team
                neighbor_team = self._generate_neighbor(current_team, bowlers, wicketkeepers)
                
                # Evaluate neighbor
                neighbor_batting_order, neighbor_batting_score = self._optimize_batting_order(neighbor_team)
                neighbor_bowling_allocation, neighbor_bowling_score = self._optimize_bowling_allocation(neighbor_team)
                
                # Calculate combined normalized score
                neighbor_score = (neighbor_batting_score / max_batting_score + 
                                 neighbor_bowling_score / max_bowling_score)
                print(neighbor_score)
                
                # Calculate acceptance probability
                delta = neighbor_score - current_score
                acceptance_probability = 1.0 if delta > 0 else np.exp(delta / temperature)
                
                # Accept or reject
                if random.random() < acceptance_probability:
                    print("Accepted")
                    current_team = neighbor_team
                    current_batting_order = neighbor_batting_order
                    current_bowling_allocation = neighbor_bowling_allocation
                    current_score = neighbor_score
                    
                    # Update best if improved
                    if current_score > best_score:
                        best_team = current_team.copy()
                        best_batting_order = current_batting_order.copy()
                        best_bowling_allocation = current_bowling_allocation.copy()
                        best_score = current_score
                        print([self.players[player_id].name for player_id in best_batting_order])
            
            # Cool down
            temperature *= cooling_rate
        
        return list(best_team), best_batting_order, best_bowling_allocation, best_score
    
    def _generate_valid_team(self, bowlers, wicketkeepers):
        """
        Generate a valid random team satisfying constraints.
        
        Args:
            bowlers: Set of player IDs who can bowl
            wicketkeepers: Set of wicketkeeper IDs
            
        Returns:
            Set of player IDs forming a valid team
        """
        team = set()
        
        # Ensure at least one wicketkeeper
        if wicketkeepers:
            keeper = random.choice(list(wicketkeepers))
            team.add(keeper)
        
        # Ensure at least 5 bowlers
        available_bowlers = list(bowlers)
        random.shuffle(available_bowlers)
        
        for bowler in available_bowlers:
            if len([p for p in team if p in bowlers]) < 5:
                team.add(bowler)
                if len(team) >= 11:
                    break
        
        # Fill remaining spots with random players
        available_players = [p for p in self.players.keys() if p not in team]
        random.shuffle(available_players)
        
        for player in available_players:
            if len(team) < 11:
                team.add(player)
            else:
                break
        
        return team
    
    def _generate_neighbor(self, current_team, bowlers, wicketkeepers):
        """
        Generate a neighboring team by swapping one or two players.
        Ensures the new team still satisfies constraints.
        
        Args:
            current_team: Current team selection
            bowlers: Set of player IDs who can bowl
            wicketkeepers: Set of wicketkeeper IDs
            
        Returns:
            New team selection (set of player IDs)
        """
        neighbor = current_team.copy()
        
        # Get players not in the team
        available_players = [p for p in self.players.keys() if p not in neighbor]
        
        if not available_players:
            return neighbor
        
        # Remove a random player
        player_to_remove = random.choice(list(neighbor))
        neighbor.remove(player_to_remove)
        
        # Add a random available player
        player_to_add = random.choice(available_players)
        neighbor.add(player_to_add)

        # Check constraints
        # 1. At least one wicketkeeper
        has_keeper = any(p in wicketkeepers for p in neighbor)
        
        # 2. At least 5 bowlers
        has_enough_bowlers = len([p for p in neighbor if p in bowlers]) >= 5
        
        # If constraints not met, undo changes
        if not (has_keeper and has_enough_bowlers):
            neighbor.remove(player_to_add)
            neighbor.add(player_to_remove)
            
            # Try to fix constraints
            if not has_keeper:
                # Replace a non-bowler with a keeper
                non_bowlers = [p for p in neighbor if p not in bowlers]
                available_keepers = [p for p in wicketkeepers if p not in neighbor]
                
                if non_bowlers and available_keepers:
                    player_to_add = random.choice(available_keepers)
                    
                    neighbor.remove(player_to_remove)
                    neighbor.add(player_to_add)
            
            elif not has_enough_bowlers:
                # Replace a non-bowler with a bowler
                non_bowlers = [p for p in neighbor if p not in bowlers]
                available_bowlers = [p for p in bowlers if p not in neighbor]
                
                if non_bowlers and available_bowlers:
                    player_to_add = random.choice(available_bowlers)
                    
                    neighbor.remove(player_to_remove)
                    neighbor.add(player_to_add)
        
        return neighbor
    
    def _optimize_batting_order(self, team):
        """
        Optimize batting order for a specific team using Hungarian algorithm.
        
        Args:
            team: Set of player IDs forming the team
            
        Returns:
            Tuple containing:
            - Optimal batting order (list of player IDs)
            - Total batting score
        """
        # Create cost matrix for Hungarian algorithm
        cost_matrix = []
        team_list = list(team)
        
        for player_id in team_list:
            player_costs = []
            for position in range(1, 12):
                # Find this player's entry for this position
                player_entry = next(
                    (p for p in self.batting_position_rankings.at[position, 'rankings'] 
                     if p['player_id'] == player_id), 
                    None
                )
                
                if player_entry:
                    # Negative because we're maximizing but Hungarian minimizes
                    weighted_score = -player_entry['score'] * self.BATTING_POSITION_WEIGHTS[position]
                    player_costs.append(weighted_score)
                else:
                    # Very high cost if player has no score for this position
                    player_costs.append(1000000)
            
            cost_matrix.append(player_costs)
        
        # Run Hungarian algorithm
        if not cost_matrix:
            return [], 0.0
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create batting order
        batting_order = [None] * 11
        total_score = 0.0
        
        for i, position_idx in enumerate(col_ind):
            if i < len(team_list):
                player_id = team_list[row_ind[i]]
                position = position_idx + 1
                
                # Find player's actual score for this position
                player_entry = next(
                    (p for p in self.batting_position_rankings.at[position, 'rankings'] 
                     if p['player_id'] == player_id), 
                    None
                )
                
                if player_entry:
                    batting_order[position_idx] = player_id
                    total_score += player_entry['score']
        
        return batting_order, total_score
    
    def _optimize_bowling_allocation(self, team):
        """
        Optimize bowling allocation for a specific team.
        
        Args:
            team: Set of player IDs forming the team
            
        Returns:
            Tuple containing:
            - Bowling allocation (dict: over -> bowler_id)
            - Total bowling score
        """
        # Get bowlers from the team
        team_bowlers = [p for p in team if any(
            p in [player['player_id'] for player in self.bowling_over_rankings.at[over, 'rankings']]
            for over in range(1, 21)
        )]
        
        # Initialize allocation and tracking variables
        allocation = {}
        overs_bowled = {bowler: 0 for bowler in team_bowlers}
        last_bowler = None
        total_score = 0.0
        
        # Loop through overs
        for over in range(1, 21):
            best_score = -1
            best_bowler = None
            
            # Check all bowlers for this over
            for player in self.bowling_over_rankings.at[over, 'rankings']:
                bowler_id = player['player_id']
                
                # Check if bowler is in the team and constraints
                if (bowler_id in team_bowlers and
                    overs_bowled.get(bowler_id, 0) < 4 and 
                    last_bowler != bowler_id):
                    
                    if player['score'] > best_score:
                        best_score = player['score']
                        best_bowler = bowler_id
            
            # Assign best bowler to this over
            if best_bowler:
                allocation[over] = best_bowler
                overs_bowled[best_bowler] = overs_bowled.get(best_bowler, 0) + 1
                last_bowler = best_bowler
                total_score += best_score
            else:
                # If no valid bowler found (due to consecutive over constraint)
                # Find any valid bowler
                for bowler_id in team_bowlers:
                    if overs_bowled.get(bowler_id, 0) < 4 and bowler_id != last_bowler:
                        # Find this bowler's score for this over
                        bowler_entry = next(
                            (p for p in self.bowling_over_rankings.at[over, 'rankings'] 
                             if p['player_id'] == bowler_id), 
                            None
                        )
                        
                        if bowler_entry:
                            allocation[over] = bowler_id
                            overs_bowled[bowler_id] = overs_bowled.get(bowler_id, 0) + 1
                            last_bowler = bowler_id
                            total_score += bowler_entry['score']
                            break
        
        return allocation, total_score
    
    def __str__(self) -> str:
        """String representation of the team."""
        return (
            f"{self.name} ({self.id}) - {len(self.players)} players, "
            f"{len(self.batsmen)} batsmen, {len(self.bowlers)} bowlers, "
            f"{len(self.all_rounders)} all-rounders, {len(self.wicket_keepers)} keepers"
        )
    
    def select_bowler(self, over, current_batsmen, match_state):
        """
        Select the best bowler for a given over based on optimal bowling allocation.
        
        Args:
            over: Current over number (1-20)
            current_batsmen: List of current batsmen IDs
            match_state: Current match state dictionary
            
        Returns:
            str: Selected bowler ID
        """
        # First check if we have an optimal allocation for this over
        if over in self.optimal_bowling_allocation:
            optimal_bowler = self.optimal_bowling_allocation[over]
            
            # Check if this bowler is still available (hasn't bowled 4 overs isn't last bowler)
            bowler_overs = match_state.get('bowler_overs', {}).get(optimal_bowler, 0)
            if bowler_overs < 4 and optimal_bowler != match_state.get('last_bowler'):
                return optimal_bowler
        
        # If optimal bowler is not available, find the next best option
        # Get all bowlers who haven't bowled 4 overs
        available_bowlers = []
        for bowler_id, overs_bowled in match_state.get('bowler_overs', {}).items():
            if overs_bowled < 4 and bowler_id != match_state.get('last_bowler'):
                # Get this bowler's ranking for the current over
                bowler_ranking = next(
                    (player for player in self.bowling_over_rankings.at[over, 'rankings']
                     if player['player_id'] == bowler_id),
                    None
                )
                if bowler_ranking:
                    available_bowlers.append((bowler_id, bowler_ranking['score']))
        
        # Sort by score and return the best available bowler
        if available_bowlers:
            available_bowlers.sort(key=lambda x: x[1], reverse=True)
            return available_bowlers[0][0]
        
        # If no bowlers are available (shouldn't happen in normal circumstances)
        # Return the first bowler from the team
        return list(self.players.keys())[0]
    
    def _optimize_team_selection(self, bowlers, wicketkeepers, max_batting_score, max_bowling_score,
                            theoretical_batting_order, theoretical_bowling_allocation):
        """
        Use branch-and-bound to find the optimal team selection.
        
        Args:
            bowlers: Set of player IDs who can bowl
            wicketkeepers: Set of wicketkeeper IDs
            max_batting_score: Theoretical maximum batting score
            max_bowling_score: Theoretical maximum bowling score
            theoretical_batting_order: Unconstrained optimal batting order
            theoretical_bowling_allocation: Unconstrained optimal bowling allocation
            
        Returns:
            Tuple containing:
            - Optimal XI (list of player IDs)
            - Optimal batting order (list of player IDs)
            - Optimal bowling allocation (dict: over -> bowler_id)
            - Combined performance score
        """
        # Initialize best solution tracking
        self.best_score = -float('inf')
        self.best_team = None
        self.best_batting_order = None
        self.best_bowling_allocation = None
        
        # Cache for memoization
        self.score_cache = {}
        
        # Sort players by individual performance for better pruning
        sorted_players = self._sort_players_by_performance()
        
        # Start branch-and-bound search
        self._branch_and_bound(
            current_team=set(),
            remaining_players=sorted_players,
            num_wicketkeepers=0,
            num_bowlers=0,
            bowlers=bowlers,
            wicketkeepers=wicketkeepers,
            max_batting_score=max_batting_score,
            max_bowling_score=max_bowling_score
        )
        
        return list(self.best_team), self.best_batting_order, self.best_bowling_allocation, self.best_score


    def _sort_players_by_performance(self):
        """
        Sort players by their individual performance metric for branch-and-bound.
        
        Returns:
            List of player IDs sorted by performance (best first)
        """
        player_scores = []
        
        for player_id, player in self.players.items():
            # Calculate individual performance score
            batting_score = 0
            bowling_score = 0
            print(player.name)
            
            # Get best batting position score for this player
            if hasattr(player, 'batting_stats'):
                for position in range(1, 12):
                    score = self._calculate_batting_position_score(player, position)
                    batting_score = max(batting_score, score * self.BATTING_POSITION_WEIGHTS[position])
                    if score>0:
                        print(position,score)
            
            # Get the sum of the 4 best bowling over scores for this player
            
            if hasattr(player, 'bowling_stats'):
                over_scores = []
                for over in range(1, 21):
                    score = self._calculate_bowling_over_score(player, over)
                    over_scores.append(score)
                    if score>0:
                        print(over, score)
                over_scores.sort()
                bowling_score = sum(over_scores[:4])

            print(batting_score, bowling_score)
            # Combined score (weighted sum)
            combined_score = np.sqrt(batting_score**2 + bowling_score**2)
            player_scores.append((player_id, player.name, combined_score))
        
        # Sort by score in descending order
        player_scores.sort(key=lambda x: x[2], reverse=True)

        print(player_scores)
        return [player_id for player_id, _, _ in player_scores]


    def _branch_and_bound(self, current_team, remaining_players, num_wicketkeepers, num_bowlers,
                        bowlers, wicketkeepers, max_batting_score, max_bowling_score):
        """
        Recursive branch-and-bound search for optimal team.
        
        Args:
            current_team: Set of currently selected player IDs
            remaining_players: List of player IDs yet to be considered
            num_wicketkeepers: Count of wicketkeepers in current team
            num_bowlers: Count of bowlers in current team
            bowlers: Set of all player IDs who can bowl
            wicketkeepers: Set of all wicketkeeper IDs
            max_batting_score: Theoretical maximum batting score
            max_bowling_score: Theoretical maximum bowling score
        """
        # Base case: team of 11 is formed
        if len(current_team) == 11:
            # Check constraints
            if num_wicketkeepers >= 1 and num_bowlers >= 5:
                # Evaluate team
                team_tuple = tuple(sorted(current_team))
                
                # Check cache first
                if team_tuple in self.score_cache:
                    score, batting_order, bowling_allocation = self.score_cache[team_tuple]
                else:
                    batting_order, batting_score = self._optimize_batting_order(current_team)
                    bowling_allocation, bowling_score = self._optimize_bowling_allocation(current_team)
                    score = (batting_score / max_batting_score + bowling_score / max_bowling_score)
                    self.score_cache[team_tuple] = (score, batting_order, bowling_allocation)
                
                # Update best if better
                if score > self.best_score:
                    self.best_score = score
                    self.best_team = current_team.copy()
                    self.best_batting_order = batting_order
                    self.best_bowling_allocation = bowling_allocation
                    print(f"New best score: {score:.4f}")
                    print([self.players[pid].name for pid in batting_order])
            return
        
        # Pruning: not enough players left
        if len(current_team) + len(remaining_players) < 11:
            return
        
        # Pruning: can't satisfy wicketkeeper constraint
        remaining_wicketkeepers = sum(1 for pid in remaining_players if pid in wicketkeepers)
        if num_wicketkeepers == 0 and remaining_wicketkeepers == 0:
            return
        
        # Pruning: can't satisfy bowler constraint
        remaining_bowlers = sum(1 for pid in remaining_players if pid in bowlers)
        if num_bowlers + remaining_bowlers < 5:
            return
        
        # Calculate upper bound
        upper_bound = self._calculate_upper_bound(
            current_team, remaining_players, max_batting_score, max_bowling_score
        )
        
        # Prune if upper bound is not promising
        if upper_bound <= self.best_score:
            return
        
        # Branch on next player
        if remaining_players:
            next_player = remaining_players[0]
            remaining = remaining_players[1:]
            
            # Branch 1: Include the player
            new_team = current_team.copy()
            new_team.add(next_player)
            new_num_wicketkeepers = num_wicketkeepers + (1 if next_player in wicketkeepers else 0)
            new_num_bowlers = num_bowlers + (1 if next_player in bowlers else 0)
            
            self._branch_and_bound(
                new_team, remaining, new_num_wicketkeepers, new_num_bowlers,
                bowlers, wicketkeepers, max_batting_score, max_bowling_score
            )
            
            # Branch 2: Exclude the player
            self._branch_and_bound(
                current_team, remaining, num_wicketkeepers, num_bowlers,
                bowlers, wicketkeepers, max_batting_score, max_bowling_score
            )


    def _calculate_upper_bound(self, current_team, remaining_players, max_batting_score, max_bowling_score):
        """
        Calculate upper bound for the current branch.
        
        Args:
            current_team: Set of currently selected player IDs
            remaining_players: List of player IDs yet to be considered
            max_batting_score: Theoretical maximum batting score
            max_bowling_score: Theoretical maximum bowling score
            
        Returns:
            float: Upper bound score for this branch
        """
        # Get current team size
        current_size = len(current_team)
        spots_left = 11 - current_size
        
        if spots_left == 0:
            # If team is complete, return actual score
            batting_order, batting_score = self._optimize_batting_order(current_team)
            bowling_allocation, bowling_score = self._optimize_bowling_allocation(current_team)
            return (batting_score / max_batting_score + bowling_score / max_bowling_score)
        
        # Estimate upper bound by adding best possible players
        # Get scores for current team members
        current_batting_scores = []
        current_bowling_scores = []
        
        for player_id in current_team:
            player = self.players[player_id]
            
            # Best batting score
            max_bat_score = 0
            for pos in range(1, 12):
                score = self._calculate_batting_position_score(player, pos)
                max_bat_score = max(max_bat_score, score * self.BATTING_POSITION_WEIGHTS[pos])
            current_batting_scores.append(max_bat_score)
            
            # Average bowling score
            if hasattr(player, 'bowling_stats'):
                total_bowl_score = 0
                overs = 0
                for over in range(1, 21):
                    score = self._calculate_bowling_over_score(player, over)
                    if score > 0:
                        total_bowl_score += score
                        overs += 1
                if overs > 0:
                    current_bowling_scores.append(total_bowl_score / overs * 4)  # Max 4 overs
        
        # Get potential scores from remaining players
        remaining_batting_scores = []
        remaining_bowling_scores = []
        
        for player_id in remaining_players[:spots_left]:  # Only consider as many as we need
            player = self.players[player_id]
            
            # Best batting score
            max_bat_score = 0
            for pos in range(1, 12):
                score = self._calculate_batting_position_score(player, pos)
                max_bat_score = max(max_bat_score, score * self.BATTING_POSITION_WEIGHTS[pos])
            remaining_batting_scores.append(max_bat_score)
            
            # Average bowling score
            if hasattr(player, 'bowling_stats'):
                total_bowl_score = 0
                overs = 0
                for over in range(1, 21):
                    score = self._calculate_bowling_over_score(player, over)
                    if score > 0:
                        total_bowl_score += score
                        overs += 1
                if overs > 0:
                    remaining_bowling_scores.append(total_bowl_score / overs * 4)
        
        # Combine and get top 11 batting scores
        all_batting_scores = current_batting_scores + remaining_batting_scores
        all_batting_scores.sort(reverse=True)
        estimated_batting_score = sum(all_batting_scores[:11])
        
        # Combine and get top bowling scores (considering max 20 overs)
        all_bowling_scores = current_bowling_scores + remaining_bowling_scores
        all_bowling_scores.sort(reverse=True)
        estimated_bowling_score = sum(all_bowling_scores[:5])  # Roughly 5 bowlers * 4 overs
        
        # Return normalized upper bound
        return (estimated_batting_score / max_batting_score + estimated_bowling_score / max_bowling_score)


    def pick_optimal_xi(self) -> Tuple[List[str], List[str], Dict[int, str]]:
        """
        Select the optimal playing XI, batting order, and bowling allocation.
        
        Returns:
            Tuple containing:
            - List of player IDs in the optimal XI
            - List of player IDs in optimal batting order
            - Dict mapping overs to bowler IDs for optimal bowling allocation
        """
        # 1. Calculate theoretical optimal batting and bowling
        # Get theoretical max batting performance
        theoretical_batting_order, max_batting_score = self._optimize_batting_order_unlimited()
        print("Theoretical batting order:")
        print([self.players[player_id].name for player_id in theoretical_batting_order])
        print(f"Max batting score: {max_batting_score}")
        
        # Get theoretical max bowling performance and allocation
        theoretical_bowling_allocation, max_bowling_score = self._optimize_bowling_allocation_unlimited()
        print("Theoretical bowling allocation:")
        print(f"Max bowling score: {max_bowling_score}")
        
        # 2. Start by identifying constraints
        # Get all players who can bowl
        bowlers = set(self.bowlers + self.all_rounders)
        
        # Get all wicketkeepers
        wicketkeepers = set(self.wicket_keepers)
        
        # 3. Search for best combination satisfying constraints using branch-and-bound
        best_xi, best_batting_order, best_bowling_allocation, best_score = self._optimize_team_selection(
            bowlers, wicketkeepers, max_batting_score, max_bowling_score, 
            theoretical_batting_order, theoretical_bowling_allocation
        )
        
        print(f"\nFinal optimal XI with score {best_score:.4f}:")
        print([self.players[pid].name for pid in best_xi])
        
        return best_xi, best_batting_order, best_bowling_allocation
    
    def pick_optimal_xi_v0(self) -> Tuple[List[str], List[str], Dict[int, str]]:
        """
        Select the optimal playing XI, batting order, and bowling allocation using brute force.
        
        Returns:
            Tuple containing:
            - List of player IDs in the optimal XI
            - List of player IDs in optimal batting order
            - Dict mapping overs to bowler IDs for optimal bowling allocation
        """
        # Get all players who can bowl
        bowlers = set(self.bowlers + self.all_rounders)

        # Get all wicketkeepers
        wicketkeepers = set(self.wicket_keepers)
        
        # Get all players
        players = set(self.players.keys())

        theoretical_batting_order, max_batting_score = self._optimize_batting_order_unlimited()
        theoretical_bowling_allocation, max_bowling_score = self._optimize_bowling_allocation_unlimited()
        
        # Generate all possible combinations of 11 players
        from itertools import combinations
        
        best_score = -float('inf')
        best_xi = None
        best_batting_order = None
        
        # Iterate through all possible combinations of 11 players with a progress bar
        total_combinations = len(list(combinations(players, 11)))
        for i, xi in enumerate(combinations(players, 11)):
            if i % 1000 == 0:
                print(f"Progress: {i/total_combinations*100:.2f}%")

            # Check if the combination satisfies the constraints
            if len(set(xi) & bowlers) >= 5 and len(set(xi) & wicketkeepers) >= 1:
                # Evaluate the team
                batting_order, batting_score = self._optimize_batting_order(xi)
                bowling_allocation, bowling_score = self._optimize_bowling_allocation(xi)
                score = (batting_score / max_batting_score + bowling_score / max_bowling_score)
                
                # Update best if score is higher
                if score > best_score:
                    best_score = score
                    best_xi = xi
                    best_batting_order = batting_order
                    best_bowling_allocation = bowling_allocation

        print(f"\nFinal optimal XI with score {best_score:.4f}:")
        print([self.players[pid].name for pid in best_xi])
        
        return best_xi, best_batting_order, best_bowling_allocation
        