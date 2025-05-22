"""T20 Cricket Match Simulator: Team Model
This module defines the Team class with objective role assignments and strategic decision-making capabilities.
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set


class Team:
    """
    Team class with objective role assignments and advanced strategy methods for cricket match simulation.
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

        # Create batting order and bowling rotation
        self.batting_order = self.create_batting_order()
        self.bowling_rotation = self.create_bowling_rotation()
    
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
    
    def create_batting_order(self, opponent=None, venue_stats=None) -> List[str]:
        """Create a batting order based on position rankings."""
        batting_order = []
        used_players = set()
        
        # Iterate through each position (1-11)
        for position in range(1, 12):
            # Get rankings for this position
            position_rankings = self.batting_position_rankings.at[position, 'rankings']
            
            # Find the best available player for this position
            for player_ranking in position_rankings:
                player_id = player_ranking['player_id']
                if player_id not in used_players:
                    batting_order.append(player_id)
                    used_players.add(player_id)
                    break
            
            # If we couldn't find a player, use any remaining player
            if len(batting_order) < position:
                for player_id in self.player_ids:
                    if player_id not in used_players:
                        batting_order.append(player_id)
                        used_players.add(player_id)
                        break
        
        return batting_order[:11]
    
    def create_bowling_rotation(self, opponent=None, venue_stats=None) -> List[str]:
        """Create a bowling rotation based on over rankings."""
        bowling_rotation = []
        used_players = set()
        overs_per_bowler = {}  # Track overs bowled by each player
        
        # Initialize overs per bowler (max 4 overs per bowler in T20)
        for player_id in self._get_available_bowlers():
            overs_per_bowler[player_id] = 0
        
        # Iterate through each over (1-20)
        for over in range(1, 21):
            # Get rankings for this over
            over_rankings = self.bowling_over_rankings.at[over, 'rankings']
            
            # Find the best available bowler who hasn't bowled 4 overs
            selected_bowler = None
            for bowler_ranking in over_rankings:
                bowler_id = bowler_ranking['player_id']
                if bowler_id not in used_players and overs_per_bowler[bowler_id] < 4:
                    selected_bowler = bowler_id
                    overs_per_bowler[bowler_id] += 1
                    if overs_per_bowler[bowler_id] == 4:
                        used_players.add(bowler_id)
                    break
            
            # If no suitable bowler found, use any available bowler
            if selected_bowler is None:
                for bowler_id in self._get_available_bowlers():
                    if overs_per_bowler[bowler_id] < 4:
                        selected_bowler = bowler_id
                        overs_per_bowler[bowler_id] += 1
                        if overs_per_bowler[bowler_id] == 4:
                            used_players.add(bowler_id)
                        break
            
            bowling_rotation.append(selected_bowler)
        
        return bowling_rotation
    
    def _get_available_bowlers(self) -> List[str]:
        """Get a list of players who can bowl."""
        return [p for p in self.players if hasattr(self.players[p], 'bowling_stats')]
    
    def _select_phase_bowlers(self, available_bowlers: List[str], min_count: int) -> List[str]:
        """Select bowlers for a specific phase of the game."""
        selected_bowlers = []
        
        # First prioritize Strike Bowlers
        strike_bowlers = [b for b in available_bowlers if 'Strike Bowler' in self.squad[b]['bowling_roles']]
        selected_bowlers.extend(strike_bowlers)
        
        # Then add Defensive Bowlers
        defensive_bowlers = [b for b in available_bowlers if 'Defensive Bowler' in self.squad[b]['bowling_roles']]
        selected_bowlers.extend([b for b in defensive_bowlers if b not in selected_bowlers])
        
        # If needed, add any remaining bowlers
        if len(selected_bowlers) < min_count:
            remaining_bowlers = [b for b in available_bowlers if b not in selected_bowlers]
            selected_bowlers.extend(remaining_bowlers)
        
        return selected_bowlers[:min_count]
    
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
        position_stats = stats.get('by_position', {}).get(str(position), {})
        
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
        
        # Calculate weighted score
        position_score = (
            position_stats.get('average', 0) * 0.3 +           # 30% weight to position average
            position_stats.get('strike_rate', 0) * 0.2 +       # 20% weight to position SR
            phase_stats.get('average', 0) * 0.2 +              # 20% weight to phase average
            phase_stats.get('strike_rate', 0) * 0.2 +          # 20% weight to phase SR
            stats.get('boundary_percentage', 0) * 0.1          # 10% weight to boundary hitting
        )
        
        return position_score
    
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
                        'position_stats': player.batting_stats.get('by_position', {}).get(str(position), {}),
                        'batting_roles': self.squad[player_id]['batting_roles']
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
        
        # Calculate weighted score (lower is better for bowling)
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
                if score > 0:  # Only include players with valid scores
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
    
    def __str__(self) -> str:
        """String representation of the team."""
        return (
            f"{self.name} ({self.id}) - {len(self.players)} players, "
            f"{len(self.batsmen)} batsmen, {len(self.bowlers)} bowlers, "
            f"{len(self.all_rounders)} all-rounders, {len(self.wicket_keepers)} keepers"
        )