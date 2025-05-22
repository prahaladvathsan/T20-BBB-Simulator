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
        1: 1.0,  # Opener
        2: 1.0,  # Opener
        3: 0.9,  # Top order
        4: 0.9,  # Top order
        5: 0.8,  # Middle order
        6: 0.8,  # Middle order
        7: 0.7,  # Finisher
        8: 0.5,  # Lower order
        9: 0.3,  # Lower order
        10: 0.2, # Tail
        11: 0.1  # Tail
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
            if player.main_role == 'batsman':
                self.batsmen.append(player_id)
            elif player.main_role == 'bowler':
                self.bowlers.append(player_id)
            elif player.main_role == 'all-rounder':
                self.all_rounders.append(player_id)
            elif player.main_role == 'wicket-keeper':
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
        
        # Get phase-specific stats based on position
        if position <= 2:  # Openers
            phase = '1'  # Powerplay
        elif position <= 5:  # Top order
            phase = '2'  # Early middle
        elif position <= 7:  # Middle order
            phase = '3'  # Late middle
        else:  # Lower order
            phase = '4'  # Death overs
            
        phase_stats = stats.get('by_phase', {}).get(phase, {})
        
        # Get overall stats
        total_runs = stats.get('total_runs', 0)
        total_balls = stats.get('total_balls', 1)  # Avoid division by zero
        total_dismissals = stats.get('dismissals', 1)  # Avoid division by zero
        
        # Calculate base stats
        overall_average = total_runs / total_dismissals if total_dismissals > 0 else 0
        overall_sr = (total_runs / total_balls * 100) if total_balls > 0 else 0
        
        # Get position-specific stats with fallbacks to overall stats
        pos_runs = position_stats.get('runs', 0)
        pos_balls = position_stats.get('balls', 0)
        pos_dismissals = position_stats.get('dismissals', 1)  # Avoid division by zero
        
        pos_average = pos_runs / pos_dismissals if pos_dismissals > 0 else overall_average
        pos_sr = (pos_runs / pos_balls * 100) if pos_balls > 0 else overall_sr
        
        # Get phase-specific stats with fallbacks
        phase_runs = phase_stats.get('runs', 0)
        phase_balls = phase_stats.get('balls', 0)
        phase_dismissals = phase_stats.get('dismissals', 1)  # Avoid division by zero
        
        phase_average = phase_runs / phase_dismissals if phase_dismissals > 0 else overall_average
        phase_sr = (phase_runs / phase_balls * 100) if phase_balls > 0 else overall_sr
        
        # Calculate weighted score with fallbacks
        position_score = (
            pos_average * 0.3 +           # 30% weight to position average
            pos_sr * 0.2 +                # 20% weight to position SR
            phase_average * 0.2 +         # 20% weight to phase average
            phase_sr * 0.2 +              # 20% weight to phase SR
            stats.get('boundary_percentage', 0) * 0.1  # 10% weight to boundary hitting
        )
        
        # Ensure the score is valid and non-negative
        return max(1.0, float(position_score))
    
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
        # Print theoretical batting order names and not IDs
        print([self.players[player_id].name for player_id in theoretical_batting_order])

        
        # Get theoretical max bowling performance and allocation
        theoretical_bowling_allocation, max_bowling_score = self._optimize_bowling_allocation_unlimited()
        
        # 2. Start by identifying constraints
        # Get all players who can bowl
        bowlers = set()
        for over in range(1, 21):
            for player in self.bowling_over_rankings.at[over, 'rankings']:
                bowlers.add(player['player_id'])
        
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
    
    def _optimize_team_selection(self, bowlers, wicketkeepers, max_batting_score, max_bowling_score,
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
                
                # Calculate acceptance probability
                delta = neighbor_score - current_score
                acceptance_probability = 1.0 if delta > 0 else np.exp(delta / temperature)
                
                # Accept or reject
                if random.random() < acceptance_probability:
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
                        print([self.players[player_id].name for player_id in best_team])
            
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
                    player_to_remove = random.choice(non_bowlers)
                    player_to_add = random.choice(available_keepers)
                    
                    neighbor.remove(player_to_remove)
                    neighbor.add(player_to_add)
            
            elif not has_enough_bowlers:
                # Replace a non-bowler with a bowler
                non_bowlers = [p for p in neighbor if p not in bowlers]
                available_bowlers = [p for p in bowlers if p not in neighbor]
                
                if non_bowlers and available_bowlers:
                    player_to_remove = random.choice(non_bowlers)
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
        
        