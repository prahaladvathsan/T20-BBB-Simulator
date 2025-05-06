"""T20 Cricket Match Simulator: Team Model
This module defines the Team class with objective role assignments and strategic decision-making capabilities.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set


class Team:
    """
    Team class with objective role assignments and advanced strategy methods for cricket match simulation.
    """
    
    # Phase definitions
    POWERPLAY = 1  # Overs 1-6
    EARLY_MIDDLE = 2  # Overs 7-12  
    LATE_MIDDLE = 3  # Overs 13-16
    DEATH = 4  # Overs 17-20
    
    PHASE_OVERS = {
        POWERPLAY: (1, 6),
        EARLY_MIDDLE: (7, 12),
        LATE_MIDDLE: (13, 16),
        DEATH: (17, 20)
    }
    
    def __init__(self, team_id: str, team_data: Dict, player_objects: Dict):
        """Initialize team with players and strategies."""
        self.id = team_id
        self.name = team_data.get('name', f"Team {team_id}")
        self.player_ids = team_data.get('players', [])
        self.players = {player_id: player_objects[player_id] for player_id in self.player_ids if player_id in player_objects}
        
        # Initialize base role lists
        self.batsmen = []
        self.bowlers = []
        self.all_rounders = []
        self.wicket_keepers = []
        
        # Initialize role dictionary with primary and secondary assignments
        self.roles = {
            'top_order': {'primary': [], 'secondary': []},
            'middle_order': {'primary': [], 'secondary': []},
            'lower_order': {'primary': [], 'secondary': []},
            'tail': {'primary': [], 'secondary': []},
            'anchor': {'primary': [], 'secondary': []},
            'pinch_hitter': {'primary': [], 'secondary': []},
            'powerplay_bowler': {'primary': [], 'secondary': []},
            'middle_overs_bowler': {'primary': [], 'secondary': []},
            'death_bowler': {'primary': [], 'secondary': []},
            'wicket_taker': {'primary': [], 'secondary': []},
            'economical_bowler': {'primary': [], 'secondary': []}
        }
        
        # Initialize game state tracking
        self.batting_order = []
        self.bowling_rotation = []
        self.current_bowler = None
        self.bowling_history = {}
        
        # Calculate team strategies based on composition
        self.strategies = self._calculate_team_strategies()
        
        # Analyze players and assign roles
        self._analyze_and_assign_roles()

    def _analyze_and_assign_roles(self):
        """Analyze player stats and assign specific roles based on objective criteria."""
        # First, categorize by main role
        for player_id, player in self.players.items():
            # Basic role categorization
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
        
        # Assign batting positions (top_order, middle_order, lower_order, tail)
        self._assign_batting_positions()
        
        # Assign batting specializations (anchor, pinch_hitter)
        self._assign_batting_specializations()
        
        # Assign bowling phase roles (powerplay, middle, death)
        self._assign_bowling_phase_roles()
        
        # Assign bowling specializations (wicket_taker, economical)
        self._assign_bowling_specializations()
        
        # For backward compatibility
        self._update_legacy_role_lists()
    
    def _assign_batting_positions(self):
        """Assign batting positions based on weighted statistics."""
        # For each player, calculate a score for each batting position role
        position_scores = {
            'top_order': {},
            'middle_order': {},
            'lower_order': {},
            'tail': {}
        }
        
        for player_id, player in self.players.items():
            if hasattr(player, 'batting_stats'):
                # TOP ORDER SCORING
                # Phase 1 (powerplay) performance is most important for top order
                phase_1_stats = player.batting_stats.get('by_phase', {}).get('1', {})
                phase_1_sr = phase_1_stats.get('strike_rate', 0)
                phase_1_avg = phase_1_stats.get('average', 0)
                
                # Calculate weighted score for top order (higher is better)
                top_order_score = (
                    phase_1_sr * 0.5 +                                       # 50% weight to powerplay SR
                    phase_1_avg * 30 * 0.3 +                                 # 30% weight to powerplay average
                    player.batting_stats.get('strike_rate', 0) * 0.1 +       # 10% weight to overall SR
                    player.batting_stats.get('average', 0) * 10 * 0.1        # 10% weight to overall average
                )
                position_scores['top_order'][player_id] = top_order_score
                
                # MIDDLE ORDER SCORING
                # Phase 2 and 3 (middle overs) performance is most important
                phase_2_stats = player.batting_stats.get('by_phase', {}).get('2', {})
                phase_3_stats = player.batting_stats.get('by_phase', {}).get('3', {})
                
                middle_sr = (
                    phase_2_stats.get('strike_rate', 0) * 0.5 + 
                    phase_3_stats.get('strike_rate', 0) * 0.5
                )
                middle_avg = (
                    phase_2_stats.get('average', 0) * 0.5 + 
                    phase_3_stats.get('average', 0) * 0.5
                )
                
                # Calculate weighted score for middle order
                middle_order_score = (
                    middle_sr * 0.4 +                                         # 40% weight to middle overs SR
                    middle_avg * 25 * 0.4 +                                   # 40% weight to middle overs average
                    player.batting_stats.get('average', 0) * 10 * 0.2         # 20% weight to overall average
                )
                position_scores['middle_order'][player_id] = middle_order_score
                
                # LOWER ORDER SCORING
                # Phase 4 (death overs) performance and finishing ability
                phase_4_stats = player.batting_stats.get('by_phase', {}).get('4', {})
                phase_4_sr = phase_4_stats.get('strike_rate', 0)
                
                # Calculate weighted score for lower order
                lower_order_score = (
                    phase_4_sr * 0.6 +                                     # 60% weight to death overs SR
                    player.batting_stats.get('strike_rate', 0) * 0.2 +     # 20% weight to overall SR
                    (player.batting_stats.get('boundary_percentage', 0) * 500) * 0.2  # 20% weight to boundary hitting
                )
                position_scores['lower_order'][player_id] = lower_order_score
                
                # TAIL SCORING - mostly based on who can hit boundaries
                # The worst batsmen are usually in the tail
                # Higher average and boundary % are the main factors
                tail_score = (
                    player.batting_stats.get('average', 0) * 5 * 0.5 +           # 50% weight to average
                    player.batting_stats.get('boundary_percentage', 0) * 300 * 0.3 +  # 30% weight to boundary hitting
                    player.batting_stats.get('strike_rate', 0) * 0.2            # 20% weight to SR
                )
                position_scores['tail'][player_id] = tail_score
        
        # For players without batting stats, assign minimal scores
        for player_id in self.players:
            if player_id not in position_scores['top_order']:
                for position in position_scores:
                    position_scores[position][player_id] = 0
        
        # Assign primary and secondary roles based on scores
        self._assign_roles_from_scores(position_scores, [3, 3, 2, 3])  # Primary: 3 top, 3 middle, 2 lower, 3 tail
    
    def _assign_batting_specializations(self):
        """Assign batting specialization roles (anchor, pinch_hitter)."""
        anchor_scores = {}
        pinch_hitter_scores = {}
        
        for player_id, player in self.players.items():
            if hasattr(player, 'batting_stats'):
                # ANCHOR SCORING - high average, good ball-per-boundary ratio, low dot percentage
                anchor_score = (
                    player.batting_stats.get('average', 0) * 15 * 0.4 +              # 40% weight to average
                    (100 - player.batting_stats.get('dot_percentage', 50)) * 3 * 0.3 +  # 30% weight to non-dot percentage
                    player.batting_stats.get('balls_per_dismissal', 0) * 0.5 * 0.3    # 30% weight to balls faced per dismissal
                )
                anchor_scores[player_id] = anchor_score
                
                # PINCH HITTER SCORING - high strike rate, high boundary percentage
                pinch_hitter_score = (
                    player.batting_stats.get('strike_rate', 0) * 0.5 +               # 50% weight to strike rate
                    player.batting_stats.get('boundary_percentage', 0) * 500 * 0.3 +  # 30% weight to boundary percentage
                    player.batting_stats.get('six_percentage', 0) * 1000 * 0.2        # 20% weight to six-hitting
                )
                pinch_hitter_scores[player_id] = pinch_hitter_score
            else:
                anchor_scores[player_id] = 0
                pinch_hitter_scores[player_id] = 0
        
        # Compile scores into a dictionary for assignment
        specialization_scores = {
            'anchor': anchor_scores,
            'pinch_hitter': pinch_hitter_scores
        }
        
        # Assign primary and secondary roles
        self._assign_roles_from_scores(specialization_scores, [2, 2])  # 2 primary anchors, 2 primary pinch hitters
    
    def _assign_bowling_phase_roles(self):
        """Assign bowling phase roles (powerplay, middle, death)."""
        powerplay_scores = {}
        middle_scores = {}
        death_scores = {}
        
        for player_id, player in self.players.items():
            if hasattr(player, 'bowling_stats'):
                # POWERPLAY BOWLING SCORE (overs 1-6)
                phase_1_stats = player.bowling_stats.get('by_phase', {}).get('1', {})
                powerplay_economy = phase_1_stats.get('economy', 99)
                powerplay_avg = phase_1_stats.get('average', 99)
                
                # Lower is better for economy, so invert the scale (10 - economy)
                powerplay_score = (
                    max(0, (10 - powerplay_economy)) * 10 * 0.5 +       # 50% weight to economy
                    max(0, (50 - powerplay_avg)) * 0.3 +               # 30% weight to average
                    phase_1_stats.get('dot_percentage', 0) * 2 * 0.2    # 20% weight to dot ball percentage
                )
                powerplay_scores[player_id] = powerplay_score
                
                # MIDDLE OVERS BOWLING SCORE (overs 7-16)
                # Combine early and late middle
                phase_2_stats = player.bowling_stats.get('by_phase', {}).get('2', {})
                phase_3_stats = player.bowling_stats.get('by_phase', {}).get('3', {})
                
                middle_economy = (
                    phase_2_stats.get('economy', 99) * 0.5 + 
                    phase_3_stats.get('economy', 99) * 0.5
                )
                
                middle_avg = (
                    phase_2_stats.get('average', 99) * 0.5 + 
                    phase_3_stats.get('average', 99) * 0.5
                )
                
                middle_score = (
                    max(0, (10 - middle_economy)) * 10 * 0.4 +     # 40% weight to economy
                    max(0, (50 - middle_avg)) * 0.4 +              # 40% weight to average
                    player.bowling_stats.get('wicket_rate', 0) * 300 * 0.2  # 20% weight to wicket rate
                )
                middle_scores[player_id] = middle_score
                
                # DEATH BOWLING SCORE (overs 17-20)
                phase_4_stats = player.bowling_stats.get('by_phase', {}).get('4', {})
                death_economy = phase_4_stats.get('economy', 99)
                
                death_score = (
                    max(0, (12 - death_economy)) * 10 * 0.6 +     # 60% weight to economy (scale adjusted for death)
                    phase_4_stats.get('dot_percentage', 0) * 2 * 0.2 +  # 20% weight to dot ball percentage
                    player.bowling_stats.get('yorker_percentage', 0) * 3 * 0.2  # 20% weight to yorker percentage
                )
                death_scores[player_id] = death_score
            else:
                powerplay_scores[player_id] = 0
                middle_scores[player_id] = 0
                death_scores[player_id] = 0
        
        # Compile scores into a dictionary for assignment
        phase_scores = {
            'powerplay_bowler': powerplay_scores,
            'middle_overs_bowler': middle_scores,
            'death_bowler': death_scores
        }
        
        # Assign primary and secondary roles
        self._assign_roles_from_scores(phase_scores, [3, 3, 3])  # 3 primary for each phase
    
    def _assign_bowling_specializations(self):
        """Assign bowling specialization roles (wicket_taker, economical)."""
        wicket_taker_scores = {}
        economical_scores = {}
        
        for player_id, player in self.players.items():
            if hasattr(player, 'bowling_stats'):
                # WICKET TAKER SCORING
                wicket_taker_score = (
                    player.bowling_stats.get('strike_rate', 0) * -0.1 + 60 * 0.4 +  # 40% weight to strike rate (lower is better)
                    player.bowling_stats.get('wicket_rate', 0) * 500 * 0.3 +      # 30% weight to wickets per over
                    player.bowling_stats.get('average', 99) * -1 + 50 * 0.3       # 30% weight to average (lower is better)
                )
                wicket_taker_scores[player_id] = max(0, wicket_taker_score)  # Ensure score is positive
                
                # ECONOMICAL BOWLER SCORING
                economical_score = (
                    max(0, (10 - player.bowling_stats.get('economy', 99))) * 20 * 0.5 +  # 50% weight to economy
                    player.bowling_stats.get('dot_percentage', 0) * 2 * 0.3 +          # 30% weight to dot percentage
                    (100 - player.bowling_stats.get('boundary_percentage', 0)) * 0.2    # 20% weight to non-boundary percentage
                )
                economical_scores[player_id] = economical_score
            else:
                wicket_taker_scores[player_id] = 0
                economical_scores[player_id] = 0
        
        # Compile scores into a dictionary for assignment
        specialization_scores = {
            'wicket_taker': wicket_taker_scores,
            'economical_bowler': economical_scores
        }
        
        # Assign primary and secondary roles
        self._assign_roles_from_scores(specialization_scores, [3, 3])  # 3 primary for each specialization
    
    def _assign_roles_from_scores(self, role_scores: Dict[str, Dict[str, float]], primary_counts: List[int]):
        """
        Assign primary and secondary roles based on calculated scores.
        
        Args:
            role_scores: Dictionary of role types with player scores for each
            primary_counts: List of how many primary players to assign for each role
        """
        for i, (role, scores) in enumerate(role_scores.items()):
            # Sort players by their score for this role (highest first)
            sorted_players = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get primary count for this role
            primary_count = primary_counts[i] if i < len(primary_counts) else 2
            secondary_count = primary_count  # Equal number of secondary roles
            
            # Assign primary roles to top scoring players
            self.roles[role]['primary'] = [player_id for player_id, score in sorted_players[:primary_count] 
                                          if score > 0]  # Only assign if score > 0
            
            # Assign secondary roles to the next set of players
            self.roles[role]['secondary'] = [player_id for player_id, score in 
                                            sorted_players[primary_count:primary_count+secondary_count] 
                                            if score > 0]  # Only assign if score > 0
    
    def _update_legacy_role_lists(self):
        """Update legacy role lists for backward compatibility."""
        # Map primary and secondary roles to legacy lists
        self.openers = self.roles['top_order']['primary'][:2]  # First 2 top order players are openers
        self.middle_order = (
            self.roles['top_order']['primary'][2:] +  # Remaining top order
            self.roles['middle_order']['primary']
        )
        self.finishers = self.roles['lower_order']['primary']
        
        self.powerplay_bowlers = self.roles['powerplay_bowler']['primary']
        self.early_middle_bowlers = self.roles['middle_overs_bowler']['primary'][:2]  # Split middle bowlers
        self.late_middle_bowlers = self.roles['middle_overs_bowler']['primary'][2:]
        self.death_bowlers = self.roles['death_bowler']['primary']
    
    def _calculate_team_strategies(self) -> Dict:
        """Calculate team strategies based on player composition and stats."""
        strategies = {}
        
        # Calculate batting aggression based on team composition
        top_order_sr = np.mean([
            self.players[p].batting_stats.get('strike_rate', 120)
            for p in self.roles['top_order']['primary']
            if hasattr(self.players[p], 'batting_stats')
        ])
        
        pinch_hitter_count = len(self.roles['pinch_hitter']['primary'])
        anchor_count = len(self.roles['anchor']['primary'])
        
        # Set batting tempo based on team strength and composition
        if top_order_sr > 140 or pinch_hitter_count >= 3:
            strategies['preferred_batting_tempo'] = 'aggressive'
            strategies['aggression_factor'] = 1.2
        elif top_order_sr > 125 or (pinch_hitter_count >= 2 and anchor_count >= 2):
            strategies['preferred_batting_tempo'] = 'balanced'
            strategies['aggression_factor'] = 1.0
        else:
            strategies['preferred_batting_tempo'] = 'conservative'
            strategies['aggression_factor'] = 0.9
        
        # Calculate bowling style based on bowler types
        spin_bowlers = len([p for p in self.bowlers + self.all_rounders
            if self.players[p].bowling_stats.get('bowling_style', '').lower()
            in ['spin', 'spinner', 'leg-spin', 'off-spin']])
        
        if spin_bowlers >= 3:
            strategies['preferred_bowling_style'] = 'spin_heavy'
        else:
            strategies['preferred_bowling_style'] = 'pace_heavy'
        
        # Dynamic bowling change threshold based on bowler quality
        death_specialist_count = len(self.roles['death_bowler']['primary'])
        economy_bowler_count = len(self.roles['economical_bowler']['primary'])
        
        if death_specialist_count >= 3 and economy_bowler_count >= 3:
            strategies['bowling_change_threshold'] = 0.8  # Less likely to change proven bowlers
        elif death_specialist_count >= 2 or economy_bowler_count >= 2:
            strategies['bowling_change_threshold'] = 0.7  # Moderate changes
        else:
            strategies['bowling_change_threshold'] = 0.6  # More willing to make changes
        
        # Powerplay strategy based on opener strength and bowling options
        if len(self.roles['powerplay_bowler']['primary']) >= 2 and len(self.roles['top_order']['primary']) >= 2:
            strategies['powerplay_strategy'] = 'aggressive'
        else:
            strategies['powerplay_strategy'] = 'conservative'
            
        return strategies
    
    def select_bowler(self, over: int, batsmen: List[str], match_state: Dict) -> str:
        """Choose a bowler for the current over based on matchups and phase."""
        # Determine current phase based on over number
        if over <= 6:
            current_phase = self.POWERPLAY
        elif over <= 12:
            current_phase = self.EARLY_MIDDLE
        elif over <= 16:
            current_phase = self.LATE_MIDDLE
        else:
            current_phase = self.DEATH
        
        # Get available bowlers (those who haven't bowled 4 overs yet)
        available_bowlers = [p for p in self.players.keys() 
                            if hasattr(self.players[p], 'bowling_stats') 
                            and match_state.get('bowler_overs', {}).get(p, 0) < 4]
        
        # Filter out bowlers who bowled the previous over (no consecutive overs)
        if 'last_bowler' in match_state and match_state['last_bowler'] in available_bowlers:
            available_bowlers.remove(match_state['last_bowler'])
        
        if not available_bowlers:
            # Emergency: allow consecutive overs
            available_bowlers = [p for p in self.players.keys() 
                                if hasattr(self.players[p], 'bowling_stats') 
                                and match_state.get('bowler_overs', {}).get(p, 0) < 4]
            
            if not available_bowlers:
                # Extreme emergency: someone has to bowl a 5th over
                available_bowlers = [p for p in self.players.keys() if hasattr(self.players[p], 'bowling_stats')]

        # Select bowler based on phase
        if current_phase == self.POWERPLAY:
            return self._select_powerplay_bowler(available_bowlers, batsmen, match_state)
        elif current_phase == self.EARLY_MIDDLE:
            return self._select_early_middle_bowler(available_bowlers, batsmen, match_state)
        elif current_phase == self.LATE_MIDDLE:
            return self._select_late_middle_bowler(available_bowlers, batsmen, match_state)
        else:  # DEATH
            return self._select_death_bowler(available_bowlers, batsmen, match_state)
    
    def __str__(self) -> str:
        """String representation of the team."""
        return f"{self.name} ({self.id}) - {len(self.players)} players, {len(self.roles['top_order']['primary'])} top order, {len(self.roles['death_bowler']['primary'])} death bowlers"