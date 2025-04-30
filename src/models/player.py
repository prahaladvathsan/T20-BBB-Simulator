"""
T20 Cricket Match Simulator: Player Model
This module defines the Player class with phase-specific and matchup-based statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class Player:
    """
    Player class with comprehensive statistics and phase-aware simulation methods.
    """
    
    # Constants for match phases
    POWERPLAY = 1  # Overs 1-6
    MIDDLE = 2     # Overs 7-15
    DEATH = 3      # Overs 16-20
    
    # Default probability distributions by phase
    PHASE_DEFAULTS = {
        POWERPLAY: {
            0: 0.33,  # dot ball
            1: 0.29,  # single
            2: 0.05,  # double
            3: 0.01,  # triple
            4: 0.18,  # four
            6: 0.08,  # six
            7: 0.01,  # wicket
            8: 0.03,  # extras (no-balls, wides)
            9: 0.02   # byes and leg-byes
        },
        MIDDLE: {
            0: 0.36,  # dot ball
            1: 0.33,  # single
            2: 0.08,  # double
            3: 0.01,  # triple
            4: 0.08,  # four
            6: 0.02,  # six
            7: 0.01,  # wicket
            8: 0.06,  # extras (no-balls, wides)
            9: 0.05   # byes and leg-byes
        },
        DEATH: {
            0: 0.25,  # dot ball
            1: 0.23,  # single
            2: 0.08,  # double
            3: 0.01,  # triple
            4: 0.15,  # four
            6: 0.15,  # six
            7: 0.04,  # wicket
            8: 0.05,  # extras (no-balls, wides)
            9: 0.04   # byes and leg-byes
        }
    }

    # Possible ball outcomes and their codes
    OUTCOMES = {
        0: "dot",
        1: "single", 
        2: "double",
        3: "triple",
        4: "four",
        6: "six",
        7: "wicket",
        8: "extras",  # Extras excluding byes (no-balls, wides)
        9: "byes"    # Byes and leg-byes
    }
    
    def __init__(self, player_id: str, player_data: Dict):
        """
        Initialize player with comprehensive statistics.
        
        Args:
            player_id: Unique identifier for the player
            player_data: Dictionary containing player information and statistics
        """
        self.id = player_id
        self.name = player_data.get('name', f"Player {player_id}")
        self.main_role = player_data.get('main_role', 'unknown')
        self.specific_roles = player_data.get('specific_roles', [])
        self.team_id = player_data.get('team_id')
        
        # Statistics
        self.batting_stats = player_data.get('batting_stats', {})
        self.bowling_stats = player_data.get('bowling_stats', {})
        
        # Derived metrics
        self._calculate_derived_metrics()
    
    def _calculate_derived_metrics(self) -> None:
        """
        Calculate derived metrics from raw statistics.
        """
        # Batting metrics
        if self.batting_stats:
            # Overall metrics
            total_runs = self.batting_stats.get('total_runs', 0)
            total_balls = self.batting_stats.get('total_balls', 0)
            self.batting_avg = total_runs / max(1, self.batting_stats.get('dismissals', 1))
            self.strike_rate = (total_runs / max(1, total_balls)) * 100
            
            # Phase-specific metrics
            self.phase_strike_rates = {}
            by_phase = self.batting_stats.get('by_phase', {})
            
            for phase, stats in by_phase.items():
                phase_runs = stats.get('runs', 0)
                phase_balls = stats.get('balls', 0)
                self.phase_strike_rates[phase] = (phase_runs / max(1, phase_balls)) * 100
            
            # Matchup metrics
            self.vs_bowler_style_stats = {}
            vs_styles = self.batting_stats.get('vs_bowler_styles', {})
            
            for style, stats in vs_styles.items():
                style_runs = stats.get('runs', 0)
                style_balls = stats.get('balls', 0)
                self.vs_bowler_style_stats[style] = {
                    'strike_rate': (style_runs / max(1, style_balls)) * 100,
                    'avg': style_runs / max(1, stats.get('dismissals', 1))
                }
            
            # Line and length preferences
            self.line_length_stats = {}
            vs_ll = self.batting_stats.get('vs_line_length', {})
            
            for ll_key, stats in vs_ll.items():
                ll_runs = stats.get('runs', 0)
                ll_balls = stats.get('balls', 0)
                self.line_length_stats[ll_key] = {
                    'strike_rate': (ll_runs / max(1, ll_balls)) * 100,
                    'avg': ll_runs / max(1, stats.get('dismissals', 1)),
                    'frequency': ll_balls / max(1, total_balls)
                }
        
        # Bowling metrics
        if self.bowling_stats:
            # Overall metrics
            total_runs_conceded = self.bowling_stats.get('runs_conceded', 0)
            total_balls_bowled = self.bowling_stats.get('balls_bowled', 0)
            total_wickets = self.bowling_stats.get('wickets', 0)
            
            self.economy_rate = (total_runs_conceded / max(1, total_balls_bowled)) * 6
            self.bowling_avg = total_runs_conceded / max(1, total_wickets)
            self.bowling_sr = total_balls_bowled / max(1, total_wickets)
            
            # Phase-specific metrics
            self.phase_economy_rates = {}
            by_phase = self.bowling_stats.get('by_phase', {})
            
            for phase, stats in by_phase.items():
                phase_runs = stats.get('runs_conceded', 0)
                phase_balls = stats.get('balls_bowled', 0)
                phase_wickets = stats.get('wickets', 0)
                
                self.phase_economy_rates[phase] = (phase_runs / max(1, phase_balls)) * 6
                # Additional phase-specific metrics can be added here
            
            # Vs batsman type metrics
            self.vs_batsman_type_stats = {}
            vs_types = self.bowling_stats.get('vs_batsman_types', {})
            
            for batter_type, stats in vs_types.items():
                type_runs = stats.get('runs_conceded', 0)
                type_balls = stats.get('balls_bowled', 0)
                type_wickets = stats.get('wickets', 0)
                
                self.vs_batsman_type_stats[batter_type] = {
                    'economy': (type_runs / max(1, type_balls)) * 6,
                    'avg': type_runs / max(1, type_wickets),
                    'sr': type_balls / max(1, type_wickets)
                }
            
            # Line and length distribution
            self.bowling_line_length = {}
            ll_dist = self.bowling_stats.get('line_length_distribution', {})
            
            for ll_key, count in ll_dist.items():
                self.bowling_line_length[ll_key] = count / max(1, total_balls_bowled)
    
    def get_batting_outcome_probability(self, bowler, phase: int, match_state: Dict) -> Tuple[Dict[int, float], int]:
        """
        Calculate probabilities for different batting outcomes.
        
        Args:
            bowler: Bowler object
            phase: Current match phase (1=Powerplay, 2=Middle, 3=Death)
            match_state: Dictionary containing current match situation
            
        Returns:
            Tuple of (probability distribution dict, number of balls the distribution is based on)
        """
        # Default probability distribution (to be used if data is missing)
        default_probs = {
            0: 0.4,  # dot ball
            1: 0.35, # single
            2: 0.1,  # double
            3: 0.01, # triple
            4: 0.1,  # four
            6: 0.03, # six
            7: 0.01, # wicket
            8: 0.01, # extras
            9: 0.01  # byes
        }
        
        # Phase-specific base probabilities
        phase_probs, phase_balls = self._get_phase_probabilities(phase)
        
        # Adjust for bowler matchup
        bowler_style = bowler.bowling_stats.get('bowling_style', 'medium')
        matchup_probs = self._adjust_for_bowler_matchup(phase_probs, bowler_style)
        
        # Adjust for match situation
        situation_probs = self._adjust_for_match_situation(matchup_probs, match_state)
        
        # Ensure probabilities sum to 1
        total_prob = sum(situation_probs.values())
        if total_prob > 0:
            normalized_probs = {k: v/total_prob for k, v in situation_probs.items()}
            return normalized_probs, phase_balls
        
        return default_probs, 0
    
    def _get_phase_probabilities(self, phase: int) -> Tuple[Dict[int, float], int]:
        """
        Get base probability distribution for the given phase.
        
        Args:
            phase: Match phase (1=Powerplay, 2=Middle, 3=Death)
            
        Returns:
            Tuple of (Dictionary of outcome probabilities, number of balls the distribution is based on)
        """
        # Default distributions by phase
        phase_defaults = self.PHASE_DEFAULTS
        
        # If we have phase-specific data for this player, use it to adjust the defaults
        phase_str = str(phase)
        if self.batting_stats and 'by_phase' in self.batting_stats and phase_str in self.batting_stats['by_phase']:
            phase_data = self.batting_stats['by_phase'][phase_str]
            
            # Extract relevant statistics
            phase_runs = phase_data.get('runs', 0)
            phase_balls = phase_data.get('balls', 0)
            phase_dots = phase_data.get('dots', 0)
            phase_fours = phase_data.get('fours', 0)
            phase_sixes = phase_data.get('sixes', 0)
            phase_dismissals = phase_data.get('dismissals', 0)
            phase_singles = phase_data.get('singles', 0)
            
            if phase_balls > 0:
                # Calculate probabilities based on historical data
                dot_prob = phase_dots / phase_balls
                four_prob = phase_fours / phase_balls
                six_prob = phase_sixes / phase_balls
                wicket_prob = phase_dismissals / phase_balls
                single_prob = phase_singles / phase_balls
                extra_prob = 0.005  # Default extras probability
                bye_prob = 0.005    # Default byes probability
                
                # Estimate remaining probabilities for 2s, 3s
                remaining_prob = 1.0 - (dot_prob + four_prob + six_prob + wicket_prob + single_prob + extra_prob + bye_prob)

                double_prob = 0.9 * remaining_prob
                triple_prob = 0.1 * remaining_prob
                
                # Create custom probability distribution
                custom_probs = {
                    0: dot_prob,
                    1: single_prob,
                    2: double_prob,
                    3: triple_prob,
                    4: four_prob,
                    6: six_prob,
                    7: wicket_prob,
                    8: extra_prob,  # Default extras probability
                    9: bye_prob     # Default byes probability
                }
                
                return custom_probs, phase_balls
        
        # Fall back to default if we don't have enough data
        return phase_defaults.get(phase, phase_defaults[self.MIDDLE]), 0
    
    def get_bowling_outcome_probability(self, batsman, phase: int, match_state: Dict) -> Tuple[Dict[int, float], int]:
        """
        Calculate probabilities for different bowling outcomes against a batsman.
        
        Args:
            batsman: Batsman object
            phase: Current match phase (1=Powerplay, 2=Middle, 3=Death)
            match_state: Dictionary containing current match situation
            
        Returns:
            Tuple of (probability distribution dict, number of balls the distribution is based on)
        """
        # Default probability distribution for bowling
        default_probs = {
            0: 0.4,  # dot ball
            1: 0.35, # single
            2: 0.1,  # double
            3: 0.01, # triple
            4: 0.1,  # four
            6: 0.03, # six
            7: 0.01, # wicket
            8: 0.01, # extras
            9: 0.01  # byes
        }
        
        # Phase-specific base probabilities
        phase_probs, phase_balls = self._get_bowling_phase_probabilities(phase)
        
        # Adjust for batsman matchup
        batsman_type = self._determine_batsman_type(batsman)
        matchup_probs = self._adjust_for_batsman_matchup(phase_probs, batsman_type)
        
        # Adjust for match situation
        situation_probs = self._adjust_bowling_for_match_situation(matchup_probs, match_state)
        
        # Ensure probabilities sum to 1
        total_prob = sum(situation_probs.values())
        if total_prob > 0:
            normalized_probs = {k: v/total_prob for k, v in situation_probs.items()}
            return normalized_probs, phase_balls
        
        return default_probs, 0

    def _get_bowling_phase_probabilities(self, phase: int) -> Tuple[Dict[int, float], int]:
        """
        Get base bowling probability distribution for the given phase.
        
        Args:
            phase: Match phase (1=Powerplay, 2=Middle, 3=Death)
            
        Returns:
            Tuple of (Dictionary of outcome probabilities, number of balls the distribution is based on)
        """
        # Default bowling distributions by phase
        phase_defaults = {
            self.POWERPLAY: {
                0: 0.35, 1: 0.3, 2: 0.05, 3: 0.01, 4: 0.2, 6: 0.08, 7: 0.01, 8: 0.005, 9: 0.005
            },
            self.MIDDLE: {
                0: 0.4, 1: 0.4, 2: 0.08, 3: 0.01, 4: 0.08, 6: 0.02, 7: 0.01, 8: 0.005, 9: 0.005
            },
            self.DEATH: {
                0: 0.3, 1: 0.25, 2: 0.1, 3: 0.01, 4: 0.15, 6: 0.15, 7: 0.04, 8: 0.01, 9: 0.01
            }
        }
        
        # If we have phase-specific data for this bowler, use it to adjust the defaults
        phase_str = str(phase)
        if self.bowling_stats and 'by_phase' in self.bowling_stats and phase_str in self.bowling_stats['by_phase']:
            phase_data = self.bowling_stats['by_phase'][phase_str]
            
            # Extract relevant statistics
            phase_runs = phase_data.get('runs_conceded', 0)
            phase_balls = phase_data.get('balls_bowled', 0)
            phase_dots = phase_data.get('dots', 0)
            phase_fours = phase_data.get('fours_conceded', 0)
            phase_sixes = phase_data.get('sixes_conceded', 0)
            phase_wickets = phase_data.get('wickets', 0)
            
            if phase_balls > 0:
                # Calculate probabilities based on historical data
                dot_prob = phase_dots / phase_balls
                four_prob = phase_fours / phase_balls
                six_prob = phase_sixes / phase_balls
                wicket_prob = phase_wickets / phase_balls
                
                # Estimate remaining probabilities for 1s, 2s, 3s
                remaining_prob = 1.0 - (dot_prob + four_prob + six_prob + wicket_prob)
                remaining_runs = phase_runs - (4 * phase_fours + 6 * phase_sixes)
                remaining_balls = phase_balls - (phase_dots + phase_fours + phase_sixes)
                
                # Distribute remaining probability
                if remaining_balls > 0:
                    avg_remaining = remaining_runs / remaining_balls
                    
                    # Simple distribution based on average
                    if avg_remaining < 1.2:
                        single_prob = 0.8 * remaining_prob
                        double_prob = 0.15 * remaining_prob
                        triple_prob = 0.05 * remaining_prob
                    elif avg_remaining < 1.8:
                        single_prob = 0.6 * remaining_prob
                        double_prob = 0.35 * remaining_prob
                        triple_prob = 0.05 * remaining_prob
                    else:
                        single_prob = 0.4 * remaining_prob
                        double_prob = 0.55 * remaining_prob
                        triple_prob = 0.05 * remaining_prob
                    
                    # Create custom probability distribution
                    custom_probs = {
                        0: dot_prob,
                        1: single_prob,
                        2: double_prob,
                        3: triple_prob,
                        4: four_prob,
                        6: six_prob,
                        7: wicket_prob,
                        8: 0.005,  # Default extras probability
                        9: 0.005   # Default byes probability
                    }
                    
                    return custom_probs, phase_balls
        
        # Fall back to default if we don't have enough data
        return phase_defaults.get(phase, phase_defaults[self.MIDDLE]), 0

    def simulate_ball_outcome(self, opponent, is_batting: bool, phase: int, match_state: Dict) -> Tuple[int, int]:
        """
        Simulate the outcome of a single ball.
        
        Args:
            opponent: Opponent Player object (bowler if is_batting=True, batsman if is_batting=False)
            is_batting: Whether this player is batting (True) or bowling (False)
            phase: Current match phase
            match_state: Current match situation
            
        Returns:
            Tuple of (outcome code, number of balls distribution is based on)
            Outcome codes: 0=dot, 1=single, 2=double, 3=triple, 4=four, 6=six, 7=wicket, 8=extras, 9=byes
        """
        if is_batting:
            # Get probabilities for batting against this bowler
            probs, balls = self.get_batting_outcome_probability(opponent, phase, match_state)
        else:
            # Get probabilities for bowling against this batsman
            probs, balls = self.get_bowling_outcome_probability(opponent, phase, match_state)
        
        # Convert probabilities to list format for numpy.random.choice
        outcomes = list(probs.keys())
        probabilities = [probs[outcome] for outcome in outcomes]
        
        # Simulate ball outcome using weighted random selection
        outcome = np.random.choice(outcomes, p=probabilities)
        
        return outcome, balls
    
    def __str__(self) -> str:
        """String representation of the player."""
        role_desc = f"{self.main_role}"
        if self.specific_roles:
            role_desc += f" ({', '.join(self.specific_roles)})"
        
        return f"{self.name} ({self.id}) - {role_desc}"