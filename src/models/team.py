"""
T20 Cricket Match Simulator: Team Model
This module defines the Team class with strategic decision-making capabilities.
"""

import random
from typing import Dict, List, Tuple, Any, Optional


class Team:
    """
    Team class with advanced strategy methods for cricket match simulation.
    """
    
    def __init__(self, team_id: str, team_data: Dict, player_objects: Dict):
        """
        Initialize team with players and strategies.
        
        Args:
            team_id: Unique identifier for the team
            team_data: Dictionary containing team information
            player_objects: Dictionary mapping player IDs to Player objects
        """
        self.id = team_id
        self.name = team_data.get('name', f"Team {team_id}")
        self.player_ids = team_data.get('players', [])
        self.players = {player_id: player_objects[player_id] for player_id in self.player_ids if player_id in player_objects}
        
        # Categorize players by role
        self._categorize_players()
        
        # Initialize batting and bowling orders
        self.batting_order = []
        self.bowling_rotation = []
        self.current_bowler = None
        self.bowling_history = {}  # Tracks overs bowled by each player
        
        # Team strategic tendencies (can be customized)
        self.strategies = {
            'aggression_factor': 1.0,  # Default neutral aggression
            'bowling_change_threshold': 0.7,  # When to consider bowling change based on performance
            'preferred_batting_tempo': 'balanced',  # balanced, aggressive, conservative
            'preferred_bowling_style': 'balanced',  # balanced, attacking, defensive
            'powerplay_strategy': 'standard'  # standard, ultra_aggressive, conservative
        }
    
    def _categorize_players(self) -> None:
        """
        Categorize players by their roles for easier selection.
        """
        # Role-based categorization
        self.batsmen = []
        self.bowlers = []
        self.all_rounders = []
        self.wicket_keepers = []
        
        # Specialized roles
        self.openers = []
        self.middle_order = []
        self.finishers = []
        self.death_bowlers = []
        self.powerplay_bowlers = []
        self.middle_overs_bowlers = []
        
        # Categorize all players
        for player_id, player in self.players.items():
            # Main role categorization
            if player.main_role == 'batsman':
                self.batsmen.append(player_id)
            elif player.main_role == 'bowler':
                self.bowlers.append(player_id)
            elif player.main_role == 'all-rounder':
                self.all_rounders.append(player_id)
            elif player.main_role == 'wicket-keeper':
                self.wicket_keepers.append(player_id)
                # Most wicket-keepers are also batsmen
                if player_id not in self.batsmen:
                    self.batsmen.append(player_id)
            
            # Specialized role categorization
            if 'opener' in player.specific_roles:
                self.openers.append(player_id)
            elif 'middle-order' in player.specific_roles:
                self.middle_order.append(player_id)
            elif 'finisher' in player.specific_roles:
                self.finishers.append(player_id)
            
            if 'death-specialist' in player.specific_roles:
                self.death_bowlers.append(player_id)
            elif 'powerplay-specialist' in player.specific_roles:
                self.powerplay_bowlers.append(player_id)
            elif 'middle-overs-specialist' in player.specific_roles:
                self.middle_overs_bowlers.append(player_id)
    
    def create_batting_order(self, opponent=None, venue_stats=None) -> List[str]:
        """
        Generate phase-optimized batting order based on player stats and match context.
        
        Args:
            opponent: Optional opponent team object
            venue_stats: Optional venue statistics
            
        Returns:
            List of player IDs in batting order
        """
        batting_order = []
        remaining_players = set(self.players.keys())
        
        # 1. Select openers (2 players)
        openers = self._select_openers(opponent, venue_stats)
        for opener in openers:
            if opener in remaining_players:
                batting_order.append(opener)
                remaining_players.remove(opener)
        
        # 2. Select middle order (typically 3-4 players)
        middle_order = self._select_middle_order(opponent, venue_stats, remaining_players)
        for player in middle_order:
            if player in remaining_players:  # Double-check to avoid duplicates
                batting_order.append(player)
                remaining_players.remove(player)
        
        # 3. Select finishers (typically 2-3 players)
        finishers = self._select_finishers(opponent, venue_stats, remaining_players)
        for player in finishers:
            if player in remaining_players:
                batting_order.append(player)
                remaining_players.remove(player)
        
        # 4. Fill remaining batting slots
        bowling_specialists = sorted(
            [p for p in remaining_players if self.players[p].main_role == 'bowler'],
            key=lambda p: self.players[p].batting_stats.get('total_runs', 0) if hasattr(self.players[p], 'batting_stats') else 0,
            reverse=True
        )
        
        batting_order.extend(bowling_specialists)
        
        # Store the batting order
        self.batting_order = batting_order
        
        return batting_order
    
    def _select_openers(self, opponent=None, venue_stats=None) -> List[str]:
        """
        Select the two opening batsmen.
        
        Args:
            opponent: Optional opponent team object
            venue_stats: Optional venue statistics
            
        Returns:
            List of player IDs for openers
        """
        selected_openers = []
        
        # Prefer designated openers
        if self.openers:
            # Sort designated openers by their powerplay strike rate if available
            sorted_openers = sorted(
                self.openers,
                key=lambda p: (
                    self.players[p].phase_strike_rates.get('1', 0) 
                    if hasattr(self.players[p], 'phase_strike_rates') 
                    else self.players[p].strike_rate
                ),
                reverse=True
            )
            
            # Take the top two openers
            selected_openers = sorted_openers[:2]
        
        # If we don't have enough designated openers, add top batsmen
        if len(selected_openers) < 2:
            # Get batsmen who aren't already selected as openers
            remaining_batsmen = [p for p in self.batsmen if p not in selected_openers]
            
            # Sort by overall strike rate
            sorted_batsmen = sorted(
                remaining_batsmen,
                key=lambda p: self.players[p].strike_rate if hasattr(self.players[p], 'strike_rate') else 0,
                reverse=True
            )
            
            # Add enough batsmen to have 2 openers total
            needed = 2 - len(selected_openers)
            selected_openers.extend(sorted_batsmen[:needed])
        
        # If we still don't have 2 openers, add all-rounders
        if len(selected_openers) < 2:
            remaining_needed = 2 - len(selected_openers)
            sorted_all_rounders = sorted(
                self.all_rounders,
                key=lambda p: self.players[p].strike_rate if hasattr(self.players[p], 'strike_rate') else 0,
                reverse=True
            )
            selected_openers.extend(sorted_all_rounders[:remaining_needed])
        
        # If somehow we still don't have 2 openers, add anyone left
        if len(selected_openers) < 2:
            remaining_players = [p for p in self.players.keys() if p not in selected_openers]
            selected_openers.extend(remaining_players[:2-len(selected_openers)])
        
        return selected_openers[:2]  # Return exactly 2 openers
    
    def _select_middle_order(self, opponent=None, venue_stats=None, remaining_players=None) -> List[str]:
        """
        Select middle order batsmen.
        
        Args:
            opponent: Optional opponent team object
            venue_stats: Optional venue statistics
            remaining_players: Set of available player IDs
            
        Returns:
            List of player IDs for middle order
        """
        if remaining_players is None:
            remaining_players = set(self.players.keys())
        
        middle_order = []
        
        # Prefer designated middle-order batsmen
        designated_middle = [p for p in self.middle_order if p in remaining_players]
        
        # Sort by average batting average (prioritize reliability)
        sorted_middle = sorted(
            designated_middle,
            key=lambda p: self.players[p].batting_avg if hasattr(self.players[p], 'batting_avg') else 0,
            reverse=True
        )
        
        # Add top 3 middle order batsmen
        middle_order.extend(sorted_middle[:3])
        
        # If we need more middle order batsmen, add strong batsmen
        if len(middle_order) < 3:
            remaining_batsmen = [p for p in self.batsmen if p in remaining_players and p not in middle_order]
            
            # Sort by batting average
            sorted_batsmen = sorted(
                remaining_batsmen,
                key=lambda p: self.players[p].batting_avg if hasattr(self.players[p], 'batting_avg') else 0,
                reverse=True
            )
            
            # Add enough batsmen to have 3 middle order batsmen total
            needed = 3 - len(middle_order)
            middle_order.extend(sorted_batsmen[:needed])
        
        # If we still need more, add all-rounders
        if len(middle_order) < 3:
            remaining_all_rounders = [p for p in self.all_rounders if p in remaining_players and p not in middle_order]
            sorted_all_rounders = sorted(
                remaining_all_rounders,
                key=lambda p: self.players[p].batting_avg if hasattr(self.players[p], 'batting_avg') else 0,
                reverse=True
            )
            
            needed = 3 - len(middle_order)
            middle_order.extend(sorted_all_rounders[:needed])
        
        return middle_order
    
    def _select_finishers(self, opponent=None, venue_stats=None, remaining_players=None) -> List[str]:
        """
        Select finishers for death overs.
        
        Args:
            opponent: Optional opponent team object
            venue_stats: Optional venue statistics
            remaining_players: Set of available player IDs
            
        Returns:
            List of player IDs for finishers
        """
        if remaining_players is None:
            remaining_players = set(self.players.keys())
        
        finishers = []
        
        # Prefer designated finishers
        designated_finishers = [p for p in self.finishers if p in remaining_players]
        
        # Sort by death overs strike rate if available
        sorted_finishers = sorted(
            designated_finishers,
            key=lambda p: (
                self.players[p].phase_strike_rates.get('3', 0) 
                if hasattr(self.players[p], 'phase_strike_rates') 
                else self.players[p].strike_rate
            ),
            reverse=True
        )
        
        # Add top 2 finishers
        finishers.extend(sorted_finishers[:2])
        
        # If we need more finishers, add hard-hitting batsmen
        if len(finishers) < 2:
            remaining_batsmen = [p for p in self.batsmen if p in remaining_players and p not in finishers]
            
            # Sort by strike rate
            sorted_batsmen = sorted(
                remaining_batsmen,
                key=lambda p: self.players[p].strike_rate if hasattr(self.players[p], 'strike_rate') else 0,
                reverse=True
            )
            
            # Add enough batsmen to have 2 finishers total
            needed = 2 - len(finishers)
            finishers.extend(sorted_batsmen[:needed])
        
        # If we still need more, add all-rounders
        if len(finishers) < 2:
            remaining_all_rounders = [p for p in self.all_rounders if p in remaining_players and p not in finishers]
            sorted_all_rounders = sorted(
                remaining_all_rounders,
                key=lambda p: self.players[p].strike_rate if hasattr(self.players[p], 'strike_rate') else 0,
                reverse=True
            )
            
            needed = 2 - len(finishers)
            finishers.extend(sorted_all_rounders[:needed])
        
        return finishers
    
    def create_bowling_rotation(self, opponent=None, venue_stats=None) -> List[str]:
        """
        Create a planned bowling rotation based on player stats and matchups.
        
        Args:
            opponent: Optional opponent team object
            venue_stats: Optional venue statistics
            
        Returns:
            List of player IDs in planned bowling rotation
        """
        # Reset bowling history for this match
        self.bowling_history = {player_id: 0 for player_id in self.players.keys()}
        
        # Identify bowling resources
        primary_bowlers = self.bowlers.copy()
        secondary_bowlers = [p for p in self.all_rounders if p not in primary_bowlers]
        part_time_bowlers = [p for p in self.batsmen if p not in primary_bowlers and p not in secondary_bowlers]
        
        # Plan is a list of (phase, bowler_id) tuples
        bowling_plan = []
        
        # Phase 1: Powerplay (overs 1-6)
        powerplay_bowlers = self._select_phase_bowlers(
            phase=1, 
            num_overs=6, 
            primary=primary_bowlers,
            secondary=secondary_bowlers,
            opponent=opponent
        )
        
        # Phase 2: Middle overs (overs 7-15)
        middle_bowlers = self._select_phase_bowlers(
            phase=2, 
            num_overs=9, 
            primary=primary_bowlers,
            secondary=secondary_bowlers,
            opponent=opponent
        )
        
        # Phase 3: Death overs (overs 16-20)
        death_bowlers = self._select_phase_bowlers(
            phase=3, 
            num_overs=5, 
            primary=primary_bowlers,
            secondary=secondary_bowlers,
            opponent=opponent
        )
        
        # Combine into a full plan
        # The actual execution may vary based on game situation
        for over in range(1, 21):
            if over <= 6:
                phase = 1
                bowlers = powerplay_bowlers
            elif over <= 15:
                phase = 2
                bowlers = middle_bowlers
            else:
                phase = 3
                bowlers = death_bowlers
            
            # Get bowler for this over from the phase plan
            phase_over = over - (0 if phase == 1 else (6 if phase == 2 else 15))
            if phase_over < len(bowlers):
                bowling_plan.append(bowlers[phase_over])
            else:
                # If we somehow ran out of planned bowlers for this phase
                available_bowlers = [p for p in primary_bowlers + secondary_bowlers if self.bowling_history.get(p, 0) < 4]
                if available_bowlers:
                    bowling_plan.append(random.choice(available_bowlers))
                else:
                    # Emergency: use part-timers if all main bowlers exhausted
                    bowling_plan.append(random.choice(part_time_bowlers) if part_time_bowlers else primary_bowlers[0])
        
        # Store the bowling rotation
        self.bowling_rotation = bowling_plan
        
        return bowling_plan
    
    def _select_phase_bowlers(self, phase: int, num_overs: int, primary: List[str], 
                             secondary: List[str], opponent=None) -> List[str]:
        """
        Select bowlers for a specific phase of the match.
        
        Args:
            phase: Match phase (1=Powerplay, 2=Middle, 3=Death)
            num_overs: Number of overs in this phase
            primary: List of primary bowler IDs
            secondary: List of secondary bowler IDs
            opponent: Optional opponent team object
            
        Returns:
            List of player IDs for selected bowlers for this phase
        """
        phase_bowlers = []
        
        # Get specialists for this phase
        if phase == 1:
            specialists = self.powerplay_bowlers
        elif phase == 2:
            specialists = self.middle_overs_bowlers
        else:
            specialists = self.death_bowlers
        
        # Combine and rank bowlers based on phase-specific economy rate
        available_bowlers = primary + secondary
        
        # Sort bowlers by phase-specific economy rate
        sorted_bowlers = sorted(
            available_bowlers,
            key=lambda p: (
                self.players[p].phase_economy_rates.get(str(phase), float('inf'))
                if hasattr(self.players[p], 'phase_economy_rates')
                else (
                    self.players[p].economy_rate 
                    if hasattr(self.players[p], 'economy_rate') 
                    else float('inf')
                )
            )
        )
        
        # Prioritize specialists but maintain a good economy
        if specialists:
            # Move specialists to the front, maintaining relative order
            sorted_specialists = [p for p in sorted_bowlers if p in specialists]
            non_specialists = [p for p in sorted_bowlers if p not in specialists]
            sorted_bowlers = sorted_specialists + non_specialists
        
        # Allocate overs for this phase
        remaining_overs = num_overs
        bowler_index = 0
        
        while remaining_overs > 0 and bowler_index < len(sorted_bowlers):
            current_bowler = sorted_bowlers[bowler_index]
            
            # Check how many overs this bowler has already been allocated
            current_allocation = self.bowling_history.get(current_bowler, 0)
            
            # Determine how many more overs this bowler can bowl
            available_overs = min(4 - current_allocation, remaining_overs)
            
            if available_overs > 0:
                # Allocate overs to this bowler
                if phase == 3:  # Death overs
                    # In death, we want our best bowlers to bowl more overs
                    overs_to_allocate = min(2, available_overs) if bowler_index < 2 else 1
                elif phase == 1:  # Powerplay
                    # In powerplay, distribute more evenly
                    overs_to_allocate = min(2, available_overs)
                else:  # Middle overs
                    # In middle overs, give slightly more to better bowlers
                    overs_to_allocate = min(3, available_overs) if bowler_index < 2 else min(2, available_overs)
                
                # Add bowler to the plan for the allocated overs
                for _ in range(overs_to_allocate):
                    phase_bowlers.append(current_bowler)
                
                # Update bowler's total overs for the match
                self.bowling_history[current_bowler] = current_allocation + overs_to_allocate
                
                # Decrease remaining overs
                remaining_overs -= overs_to_allocate
            
            # Move to the next bowler
            bowler_index += 1
        
        # If we somehow still have overs to allocate, distribute them
        if remaining_overs > 0:
            available_bowlers = [p for p in sorted_bowlers if self.bowling_history.get(p, 0) < 4]
            
            while remaining_overs > 0 and available_bowlers:
                selected_bowler = available_bowlers.pop(0)
                phase_bowlers.append(selected_bowler)
                self.bowling_history[selected_bowler] = self.bowling_history.get(selected_bowler, 0) + 1
                remaining_overs -= 1
        
        return phase_bowlers
    
    def select_bowler(self, over: int, batsmen: List[str], match_state: Dict) -> str:
        """
        Choose a bowler for the current over based on matchups and phase.
        
        Args:
            over: Current over number
            batsmen: List of current batsmen player IDs
            match_state: Dictionary with match situation
            
        Returns:
            Selected bowler's player ID
        """
        # Determine current phase
        if over <= 6:
            current_phase = 1  # Powerplay
        elif over <= 15:
            current_phase = 2  # Middle overs
        else:
            current_phase = 3  # Death overs
        
        # Get available bowlers (those who haven't bowled 4 overs yet)
        available_bowlers = [p for p in self.players.keys() 
                            if hasattr(self.players[p], 'bowling_stats') 
                            and match_state.get('bowler_overs', {}).get(p, 0) < 4]
        
        # Filter out bowlers who bowled the previous over (no consecutive overs)
        if 'last_bowler' in match_state and match_state['last_bowler'] in available_bowlers:
            available_bowlers.remove(match_state['last_bowler'])
        
        if not available_bowlers:
            # Emergency situation - all bowlers have bowled 4 overs or consecutive
            # Allow consecutive overs in this emergency
            available_bowlers = [p for p in self.players.keys() 
                                if hasattr(self.players[p], 'bowling_stats') 
                                and match_state.get('bowler_overs', {}).get(p, 0) < 4]
            
            if not available_bowlers:
                # Extreme emergency - someone has to bowl a 5th over
                # Choose the best economy bowler for the current phase
                all_bowlers = [p for p in self.players.keys() if hasattr(self.players[p], 'bowling_stats')]
                available_bowlers = all_bowlers
        
        # Different selection strategies based on phase and match state
        if current_phase == 3:  # Death overs
            # In death, prioritize death specialists and good matchups
            return self._select_death_bowler(available_bowlers, batsmen, match_state)
            
        elif current_phase == 1:  # Powerplay
            # In powerplay, prioritize powerplay specialists
            return self._select_powerplay_bowler(available_bowlers, batsmen, match_state)
            
        else:  # Middle overs
            # In middle overs, focus on matchups and maintaining pressure
            return self._select_middle_overs_bowler(available_bowlers, batsmen, match_state)
    
    def _select_death_bowler(self, available_bowlers: List[str], batsmen: List[str], match_state: Dict) -> str:
        """
        Select a bowler for death overs.
        
        Args:
            available_bowlers: List of available bowler IDs
            batsmen: List of current batsmen IDs
            match_state: Current match state
            
        Returns:
            Selected bowler's player ID
        """
        # Prioritize death specialists
        death_specialists = [p for p in available_bowlers if p in self.death_bowlers]
        
        if death_specialists:
            # Choose the specialist with the best death economy
            best_specialist = min(
                death_specialists,
                key=lambda p: (
                    self.players[p].phase_economy_rates.get('3', float('inf'))
                    if hasattr(self.players[p], 'phase_economy_rates')
                    else self.players[p].economy_rate if hasattr(self.players[p], 'economy_rate') else float('inf')
                )
            )
            return best_specialist
        
        # If no death specialists available, use economy rate in death overs
        sorted_by_economy = sorted(
            available_bowlers,
            key=lambda p: (
                self.players[p].phase_economy_rates.get('3', float('inf'))
                if hasattr(self.players[p], 'phase_economy_rates')
                else self.players[p].economy_rate if hasattr(self.players[p], 'economy_rate') else float('inf')
            )
        )
        
        # Get the best matchup against current batsmen
        batsman_player = self.players.get(batsmen[0]) if batsmen else None
        
        if batsman_player:
            batsman_type = 'aggressive' if (hasattr(batsman_player, 'strike_rate') and batsman_player.strike_rate > 150) else 'standard'
            
            # Check if any bowler has a good matchup against this type of batsman
            for bowler_id in sorted_by_economy[:3]:  # Consider top 3 economy bowlers
                bowler = self.players[bowler_id]
                if (hasattr(bowler, 'vs_batsman_type_stats') and 
                    batsman_type in bowler.vs_batsman_type_stats and
                    bowler.vs_batsman_type_stats[batsman_type].get('economy', float('inf')) < bowler.economy_rate):
                    return bowler_id
        
        # Default to best economy bowler if no good matchups
        return sorted_by_economy[0] if sorted_by_economy else available_bowlers[0]
    
    def _select_powerplay_bowler(self, available_bowlers: List[str], batsmen: List[str], match_state: Dict) -> str:
        """
        Select a bowler for powerplay overs.
        
        Args:
            available_bowlers: List of available bowler IDs
            batsmen: List of current batsmen IDs
            match_state: Current match state
            
        Returns:
            Selected bowler's player ID
        """
        # Prioritize powerplay specialists
        powerplay_specialists = [p for p in available_bowlers if p in self.powerplay_bowlers]
        
        if powerplay_specialists:
            # Choose the specialist with the best powerplay economy
            best_specialist = min(
                powerplay_specialists,
                key=lambda p: (
                    self.players[p].phase_economy_rates.get('1', float('inf'))
                    if hasattr(self.players[p], 'phase_economy_rates')
                    else self.players[p].economy_rate if hasattr(self.players[p], 'economy_rate') else float('inf')
                )
            )
            return best_specialist
        
        # If no powerplay specialists available, use economy rate in powerplay
        sorted_by_economy = sorted(
            available_bowlers,
            key=lambda p: (
                self.players[p].phase_economy_rates.get('1', float('inf'))
                if hasattr(self.players[p], 'phase_economy_rates')
                else self.players[p].economy_rate if hasattr(self.players[p], 'economy_rate') else float('inf')
            )
        )
        
        # Check specifically for opening batsmen
        is_opener = match_state.get('over', 0) <= 2  # First two overs
        
        if is_opener:
            # Prioritize bowlers who do well against openers
            for bowler_id in sorted_by_economy[:3]:  # Consider top 3 economy bowlers
                bowler = self.players[bowler_id]
                if (hasattr(bowler, 'vs_batsman_type_stats') and 
                    'opener' in bowler.vs_batsman_type_stats and
                    bowler.vs_batsman_type_stats['opener'].get('economy', float('inf')) < bowler.economy_rate):
                    return bowler_id
        
        # Default to best economy bowler if no good matchups
        return sorted_by_economy[0] if sorted_by_economy else available_bowlers[0]
    
    def _select_middle_overs_bowler(self, available_bowlers: List[str], batsmen: List[str], match_state: Dict) -> str:
        """
        Select a bowler for middle overs.
        
        Args:
            available_bowlers: List of available bowler IDs
            batsmen: List of current batsmen IDs
            match_state: Current match state
            
        Returns:
            Selected bowler's player ID
        """
        # Prioritize middle overs specialists
        middle_specialists = [p for p in available_bowlers if p in self.middle_overs_bowlers]
        
        # Get spinners (often effective in middle overs)
        spinners = [p for p in available_bowlers 
                   if self.players[p].bowling_stats.get('bowling_style', '').lower() in ['spin', 'spinner', 'leg-spin', 'off-spin']]
        
        middle_candidates = []
        
        # Combine specialists and spinners, prioritizing those who are both
        for p in available_bowlers:
            if p in middle_specialists and p in spinners:
                middle_candidates.insert(0, p)  # Highest priority
            elif p in middle_specialists:
                middle_candidates.append(p)  # Medium priority
            elif p in spinners:
                middle_candidates.append(p)  # Medium priority
        
        # Add remaining bowlers
        for p in available_bowlers:
            if p not in middle_candidates:
                middle_candidates.append(p)  # Lowest priority
        
        # Sort candidates by economy in middle overs
        sorted_candidates = sorted(
            middle_candidates,
            key=lambda p: (
                self.players[p].phase_economy_rates.get('2', float('inf'))
                if hasattr(self.players[p], 'phase_economy_rates')
                else self.players[p].economy_rate if hasattr(self.players[p], 'economy_rate') else float('inf')
            )
        )
        
        # Consider partnerships and current batsmen
        # If batsmen have been scoring well, consider introducing a different bowling style
        if match_state.get('current_partnership_runs', 0) > 30:
            # Look for a bowler with a different style than recent overs
            recent_style = match_state.get('recent_bowling_style', None)
            
            if recent_style:
                for bowler_id in sorted_candidates:
                    bowler_style = self.players[bowler_id].bowling_stats.get('bowling_style', '')
                    if bowler_style.lower() != recent_style.lower():
                        return bowler_id
        
        # Default to best economy bowler
        return sorted_candidates[0] if sorted_candidates else available_bowlers[0]
    
    def adjust_strategy(self, match_state: Dict) -> Dict:
        """
        Modify approach based on match situation.
        
        Args:
            match_state: Current match state dictionary
            
        Returns:
            Updated strategies dictionary
        """
        updated_strategies = self.strategies.copy()
        
        # Batting adjustments
        if match_state.get('is_batting', False):
            # Batting in first innings
            if match_state.get('innings', 1) == 1:
                # Early overs strategy
                if match_state.get('over', 0) <= 6:
                    if self.strategies['preferred_batting_tempo'] == 'aggressive':
                        updated_strategies['aggression_factor'] = 1.3
                    elif self.strategies['preferred_batting_tempo'] == 'conservative':
                        updated_strategies['aggression_factor'] = 0.8
                # Middle overs strategy
                elif match_state.get('over', 0) <= 15:
                    # If we've lost too many wickets, become more conservative
                    if match_state.get('wickets', 0) >= 4:
                        updated_strategies['aggression_factor'] = 0.7
                # Death overs strategy
                else:
                    # In death overs, increase aggression
                    updated_strategies['aggression_factor'] = 1.5
                    
                    # But be more careful if we've lost many wickets
                    if match_state.get('wickets', 0) >= 7:
                        updated_strategies['aggression_factor'] = 1.0
                    
            # Batting in second innings (chasing)
            else:
                required_run_rate = match_state.get('required_run_rate', 6.0)
                remaining_balls = match_state.get('balls_remaining', 0)
                
                # Easy chase
                if required_run_rate < 7.0:
                    updated_strategies['aggression_factor'] = 0.9
                # Moderate chase
                elif required_run_rate < 10.0:
                    updated_strategies['aggression_factor'] = 1.1
                # Difficult chase
                elif required_run_rate < 14.0:
                    updated_strategies['aggression_factor'] = 1.3
                # Very difficult chase
                else:
                    updated_strategies['aggression_factor'] = 1.5
                
                # Final overs of a close chase
                if remaining_balls <= 30 and required_run_rate > 9.0:
                    updated_strategies['aggression_factor'] = 1.6
        
        # Bowling adjustments
        else:
            # Defending in second innings
            if match_state.get('innings', 1) == 2:
                required_run_rate = match_state.get('required_run_rate', 6.0)
                
                # If run rate is high, play more attacking
                if required_run_rate > 12.0:
                    updated_strategies['preferred_bowling_style'] = 'attacking'
                # If run rate is low, be more defensive
                elif required_run_rate < 8.0:
                    updated_strategies['preferred_bowling_style'] = 'defensive'
            
            # Adjust bowling change threshold based on match state
            if match_state.get('recent_boundary_count', 0) >= 3:
                # Lower threshold = more likely to change bowlers
                updated_strategies['bowling_change_threshold'] = 0.4
            elif match_state.get('recent_dot_count', 0) >= 6:
                # Higher threshold = less likely to change bowlers
                updated_strategies['bowling_change_threshold'] = 0.9
        
        # Update strategies
        self.strategies = updated_strategies
        return updated_strategies
    
    def __str__(self) -> str:
        """String representation of the team."""
        return f"{self.name} ({self.id}) - {len(self.players)} players"