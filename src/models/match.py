"""match_get_results
T20 Cricket Match Simulator: Match Model
This module defines the Match class for detailed match simulation.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class Match:
    """
    Match class with detailed simulation logic for T20 cricket matches.
    """
    
    def __init__(self, team1, team2, venue_stats=None):
        """
        Initialize a cricket match between two teams.
        
        Args:
            team1: First Team object
            team2: Second Team object
            venue_stats: Optional venue statistics
        """
        self.team1 = team1
        self.team2 = team2
        self.venue_stats = venue_stats or {}
        
        # Match state
        self.toss_winner = None
        self.toss_decision = None
        self.batting_first = None
        self.bowling_first = None
        self.current_innings = 0
        self.match_completed = False
        self.winner = None
        
        # Innings data
        self.innings_data = {
            1: self._create_empty_innings(),
            2: self._create_empty_innings()
        }
        
        # Match events log
        self.events = []
        
        # Key moments tracking
        self.key_moments = []
        
        # Win probability progression
        self.win_probability = []
    
    def _create_empty_innings(self) -> Dict:
        """
        Create an empty innings data structure.
        
        Returns:
            Dictionary with initialized innings data
        """
        return {
            'batting_team': None,
            'bowling_team': None,
            'score': 0,
            'wickets': 0,
            'overs': 0,
            'balls': 0,
            'extras': 0,
            'batting_order': [],
            'current_batsmen': [],
            'dismissed': [],
            'extras_detail': {'wide': 0, 'no_ball': 0, 'bye': 0, 'leg_bye': 0},
            'bowlers_used': [],
            'current_bowler': None,
            'partnerships': [],
            'current_partnership': {'runs': 0, 'balls': 0},
            'phase_scores': {1: 0, 2: 0, 3: 0, 4: 0},
            'phase_wickets': {1: 0, 2: 0, 3: 0, 4: 0},
            'phase_balls': {1: 0, 2: 0, 3: 0, 4: 0},
            'over_history': {},
            'batsmen_stats': {},
            'bowler_stats': {}
        }
    
    def simulate_match(self) -> Dict:
        """
        Simulate an entire T20 cricket match.
        
        Returns:
            Dictionary with match results and statistics
        """
        # 1. Simulate toss
        self.simulate_toss()
        
        # Log toss result
        self.events.append({
            'event_type': 'toss',
            'winner': self.toss_winner.name,
            'decision': self.toss_decision
        })
        
        # 2. Set up batting orders and bowling plans
        self.batting_first.create_batting_order(opponent=self.bowling_first, venue_stats=self.venue_stats)
        self.bowling_first.create_bowling_rotation(opponent=self.batting_first, venue_stats=self.venue_stats)
        
        # 3. Simulate first innings
        self.current_innings = 1
        self.innings_data[1]['batting_team'] = self.batting_first
        self.innings_data[1]['bowling_team'] = self.bowling_first
        self.innings_data[1]['batting_order'] = self.batting_first.batting_order.copy()
        first_innings_result = self.simulate_innings()
        
        # Log innings completion
        self.events.append({
            'event_type': 'innings_complete',
            'innings': 1,
            'score': self.innings_data[1]['score'],
            'wickets': self.innings_data[1]['wickets'],
            'overs': self.innings_data[1]['overs'],
            'balls': self.innings_data[1]['balls']
        })
        
        # Switch batting/bowling teams
        self.bowling_first.create_batting_order(opponent=self.batting_first, venue_stats=self.venue_stats)
        self.batting_first.create_bowling_rotation(opponent=self.bowling_first, venue_stats=self.venue_stats)
        
        # 4. Simulate second innings
        self.current_innings = 2
        self.innings_data[2]['batting_team'] = self.bowling_first
        self.innings_data[2]['bowling_team'] = self.batting_first
        self.innings_data[2]['batting_order'] = self.bowling_first.batting_order.copy()
        
        # Set target
        target = self.innings_data[1]['score'] + 1
        self.innings_data[2]['target'] = target
        
        second_innings_result = self.simulate_innings(target=target)
        
        # Log innings and match completion
        self.events.append({
            'event_type': 'innings_complete',
            'innings': 2,
            'score': self.innings_data[2]['score'],
            'wickets': self.innings_data[2]['wickets'],
            'overs': self.innings_data[2]['overs'],
            'balls': self.innings_data[2]['balls']
        })
        
        # 5. Determine the winner
        self.match_completed = True
        self.determine_winner()
        
        self.events.append({
            'event_type': 'match_complete',
            'winner': self.winner.name if self.winner else 'Tie',
            'margin': self._get_victory_margin()
        })
        
        # 6. Prepare and return match results
        return self.get_match_results()
    
    def simulate_toss(self) -> Tuple[Any, str]:
        """
        Simulate the toss and decision.
        
        Returns:
            Tuple of (toss winner, decision)
        """
        # Randomly determine toss winner
        self.toss_winner = random.choice([self.team1, self.team2])
        loser = self.team2 if self.toss_winner == self.team1 else self.team1
        
        # Determine toss decision based on venue statistics if available
        bat_first_win_pct = self.venue_stats.get('first_innings_win_percentage', 50)
        
        # Teams generally prefer chasing in T20s unless conditions strongly favor batting first
        bat_first_preference = 40  # Base preference percentage
        
        # Adjust based on venue statistics
        if bat_first_win_pct > 60:
            bat_first_preference = 75  # Strong preference for batting first
        elif bat_first_win_pct > 50:
            bat_first_preference = 55  # Slight preference for batting first
        
        # Make the decision
        if random.randint(1, 100) <= bat_first_preference:
            self.toss_decision = 'bat'
            self.batting_first = self.toss_winner
            self.bowling_first = loser
        else:
            self.toss_decision = 'field'
            self.batting_first = loser
            self.bowling_first = self.toss_winner
        
        return self.toss_winner, self.toss_decision
    
    def simulate_innings(self, target=None) -> Dict:
        """
        Simulate one innings of a T20 match.
        
        Args:
            target: Optional target score for the second innings
            
        Returns:
            Dictionary with innings results
        """
        innings_data = self.innings_data[self.current_innings]
        batting_team = innings_data['batting_team']
        bowling_team = innings_data['bowling_team']
        
        # Initialize batsmen - first two in the batting order
        if len(innings_data['batting_order']) >= 2:
            striker_id = innings_data['batting_order'][0]
            non_striker_id = innings_data['batting_order'][1]
            
            innings_data['current_batsmen'] = [striker_id, non_striker_id]
            innings_data['batsmen_stats'][striker_id] = {
                'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'dots': 0, 'strike_rate': 0
            }
            innings_data['batsmen_stats'][non_striker_id] = {
                'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'dots': 0, 'strike_rate': 0
            }

            # Remove batsmen from batting order as they're now active
            innings_data['batting_order'] = innings_data['batting_order'][2:]
        
        # Initialize partnership
        innings_data['current_partnership'] = {'runs': 0, 'balls': 0}
        
        # Simulate each over
        for over in range(1, 21):
            # Check if innings is already over (all out or target reached)
            if innings_data['wickets'] >= 10 or (target and innings_data['score'] >= target):
                break
            
            # Set current phase
            current_phase = 1 if over <= 6 else (2 if over <= 12 else (3 if over <= 16 else 4))
            
            # Update match state for bowler selection
            match_state = self._get_current_match_state()
            
            # Select bowler for this over
            bowler_id = bowling_team.select_bowler(over, innings_data['current_batsmen'], match_state)
            innings_data['current_bowler'] = bowler_id
            
            if bowler_id not in innings_data['bowlers_used']:
                innings_data['bowlers_used'].append(bowler_id)
                innings_data['bowler_stats'][bowler_id] = {
                    'overs': 0, 'balls': 0, 'maidens': 0, 'runs': 0, 'wickets': 0, 'economy': 0
                }
            
            # Initialize over statistics
            over_runs = 0
            over_wickets = 0
            over_extras = 0
            over_balls = 0
            
            # Simulate each ball in the over
            for ball in range(1, 7):
                # Check if innings is already over
                if innings_data['wickets'] >= 10 or (target and innings_data['score'] >= target):
                    break
                
                # Get current batsmen and bowler objects
                striker_id = innings_data['current_batsmen'][0]
                striker = batting_team.players[striker_id]
                bowler = bowling_team.players[bowler_id]
                
                # Update match state for ball simulation
                match_state = self._get_current_match_state()
                match_state['current_phase'] = current_phase
                
                # Simulate ball outcome
                outcome = self._simulate_ball(striker, bowler, current_phase, match_state)
                
                # Process the outcome
                ball_runs, is_wicket, is_extra, extras_type = self._process_ball_outcome(outcome, innings_data)
                
                # Update innings statistics
                innings_data['score'] += ball_runs
                over_runs += ball_runs
                
                if is_wicket:
                    innings_data['wickets'] += 1
                    over_wickets += 1
                    innings_data['dismissed'].append(striker_id)
                    
                    # Log wicket event
                    self.events.append({
                        'event_type': 'wicket',
                        'innings': self.current_innings,
                        'over': over,
                        'ball': ball,
                        'batsman': striker.name,
                        'bowler': bowler.name,
                        'wicket_type': 'bowled',  # Simplified, could be more complex
                        'score': f"{innings_data['score']}/{innings_data['wickets']}"
                    })
                    
                    # End partnership and start new one
                    innings_data['partnerships'].append(innings_data['current_partnership'])
                    innings_data['current_partnership'] = {'runs': 0, 'balls': 0}
                    
                    # Bring in new batsman if available
                    if innings_data['batting_order']:
                        new_batsman_id = innings_data['batting_order'].pop(0)
                        innings_data['current_batsmen'][0] = new_batsman_id
                        innings_data['batsmen_stats'][new_batsman_id] = {
                            'runs': 0, 'balls': 0, 'fours': 0, 'sixes': 0, 'dots': 0, 'strike_rate': 0
                        }
                    else:
                        # No more batsmen, innings is over
                        break
                
                if not is_extra:  # Regular delivery
                    innings_data['balls'] += 1
                    over_balls += 1
                    
                    # Update batsman's stats
                    innings_data['batsmen_stats'][striker_id]['balls'] += 1
                    if ball_runs > 0:
                        innings_data['batsmen_stats'][striker_id]['runs'] += ball_runs
                        if ball_runs == 4:
                            innings_data['batsmen_stats'][striker_id]['fours'] += 1
                        elif ball_runs == 6:
                            innings_data['batsmen_stats'][striker_id]['sixes'] += 1
                    else:
                        innings_data['batsmen_stats'][striker_id]['dots'] += 1
                    
                    # Update strike rate
                    balls = innings_data['batsmen_stats'][striker_id]['balls']
                    runs = innings_data['batsmen_stats'][striker_id]['runs']
                    innings_data['batsmen_stats'][striker_id]['strike_rate'] = (runs / balls) * 100 if balls > 0 else 0
                    
                    # Update bowler's stats
                    innings_data['bowler_stats'][bowler_id]['balls'] += 1
                    innings_data['bowler_stats'][bowler_id]['runs'] += ball_runs
                    if is_wicket:
                        innings_data['bowler_stats'][bowler_id]['wickets'] += 1
                    
                    # Update economy rate
                    bowler_balls = innings_data['bowler_stats'][bowler_id]['balls']
                    bowler_runs = innings_data['bowler_stats'][bowler_id]['runs']
                    innings_data['bowler_stats'][bowler_id]['economy'] = (bowler_runs / bowler_balls) * 6 if bowler_balls > 0 else 0
                    
                    # Update partnership
                    innings_data['current_partnership']['runs'] += ball_runs
                    innings_data['current_partnership']['balls'] += 1
                    
                    # Update phase statistics
                    innings_data['phase_scores'][current_phase] += ball_runs
                    if is_wicket:
                        innings_data['phase_wickets'][current_phase] += 1
                    innings_data['phase_balls'][current_phase] += 1
                    
                    # Rotate strike for odd number of runs (excluding wicket deliveries)
                    if ball_runs % 2 == 1 and not is_wicket:
                        innings_data['current_batsmen'].reverse()
                else:
                    # Handle extras
                    innings_data['extras'] += ball_runs
                    innings_data['extras_detail'][extras_type] += ball_runs
                    over_extras += ball_runs
                    
                    # Some extras don't count as ball faced but add to the score
                    if extras_type in ['wide', 'no_ball']:
                        ball -= 1  # No change in balls count for these extras
                    else:
                        # Byes and leg byes count as a ball faced
                        innings_data['balls'] += 1
                        over_balls += 1
                        innings_data['current_partnership']['balls'] += 1
                
                # Log boundary events
                if ball_runs >= 4 and not is_extra:
                    self.events.append({
                        'event_type': 'boundary',
                        'innings': self.current_innings,
                        'over': over,
                        'ball': ball,
                        'batsman': striker.name,
                        'bowler': bowler.name,
                        'runs': ball_runs,
                        'score': f"{innings_data['score']}/{innings_data['wickets']}"
                    })
                
                # Calculate and update win probability
                self._update_win_probability()
                
                # Check for key moments
                self._check_for_key_moment(over, ball, innings_data, ball_runs, is_wicket)
            
            # End of over - rotate strike
            innings_data['current_batsmen'].reverse()
            
            # Update overs count
            innings_data['overs'] += 1
            
            # Check if it was a maiden over
            if over_runs == 0 and over_balls == 6:
                innings_data['bowler_stats'][bowler_id]['maidens'] += 1
            
            # Update bowler's over count
            innings_data['bowler_stats'][bowler_id]['overs'] += over_balls / 6
            
            # Store over history
            innings_data['over_history'][over] = {
                'runs': over_runs,
                'wickets': over_wickets,
                'extras': over_extras,
                'bowler': bowler_id
            }
            
            # Update team strategies based on match situation
            batting_team.adjust_strategy(self._get_current_match_state(is_batting=True))
            bowling_team.adjust_strategy(self._get_current_match_state(is_batting=False))
        
        # End of innings
        return innings_data
    
    def _simulate_ball(self, batsman, bowler, phase, match_state):
        """
        Simulate the outcome of a single ball.
        
        Args:
            batsman: Batsman Player object
            bowler: Bowler Player object
            phase: Current match phase
            match_state: Current match situation
            
        Returns:
            Outcome code (int) representing the ball result
        """
        # Get probabilities from both batsman and bowler
        batsman_prob = batsman.get_batting_outcome_probability(bowler, phase, match_state)
        bowler_prob = bowler.get_bowling_outcome_probability(batsman, phase, match_state)
        
        # Combine probabilities with weighted average
        # Giving slightly more weight to the batsman in T20 format
        combined_probs = {}
        for outcome in set(list(batsman_prob.keys()) + list(bowler_prob.keys())):
            batsman_value = batsman_prob.get(outcome, 0)
            bowler_value = bowler_prob.get(outcome, 0)
            combined_probs[outcome] = 0.55 * batsman_value + 0.45 * bowler_value
        
        # Convert probabilities to list format for random.choices
        outcomes = list(combined_probs.keys())
        probabilities = [combined_probs[outcome] for outcome in outcomes]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        normalized_probs = [p/total_prob for p in probabilities]
        
        # Select outcome based on probabilities
        return random.choices(outcomes, weights=normalized_probs, k=1)[0]
    
    def _process_ball_outcome(self, outcome, innings_data):
        """
        Process the outcome of a simulated ball.
        
        Args:
            outcome: Ball outcome code
            innings_data: Current innings data
            
        Returns:
            Tuple of (runs scored, is wicket, is extra, extras type)
        """
        # Default return values
        runs = 0
        is_wicket = False
        is_extra = False
        extras_type = None
        
        # Process based on outcome code
        if outcome == 0:  # dot ball
            runs = 0
        elif outcome in [1, 2, 3, 4, 6]:  # regular runs
            runs = outcome
        elif outcome == 7:  # wicket
            is_wicket = True
        elif outcome == 8:  # extras (wides/no-balls)
            is_extra = True
            # Randomly choose between wide and no-ball
            extras_type = random.choice(['wide', 'no_ball'])
            runs = 1
            # Additional runs on extras (rare but possible)
            if random.random() < 0.40:  # 15% chance of additional runs
                runs += random.choice([1, 2, 4, 6])
        elif outcome == 9:  # byes/leg-byes
            is_extra = True
            # Randomly choose between bye and leg-bye
            extras_type = random.choice(['bye', 'leg_bye'])
            # Byes typically result in 1-4 runs
            runs = random.choices([1, 2, 3, 4], weights=[0.6, 0.1, 0.05, 0.25], k=1)[0]

        return runs, is_wicket, is_extra, extras_type
    
    def _get_current_match_state(self, is_batting=None):
        """
        Get the current match state for decision-making.
        
        Args:
            is_batting: Optional boolean to specify batting/bowling perspective
            
        Returns:
            Dictionary with current match state
        """
        innings_data = self.innings_data[self.current_innings]
        
        # Determine current phase
        over = innings_data['overs'] + 1  # Current over (1-indexed)
        if over <= 6:
            current_phase = 1  # Powerplay
        elif over <= 12:
            current_phase = 2  # Early Middle
        elif over <= 16:
            current_phase = 3  # Late Middle
        else:
            current_phase = 4  # Death
        
        # Calculate balls remaining
        total_balls = 120
        balls_bowled = innings_data['overs'] * 6 + (innings_data['balls'] % 6)
        balls_remaining = total_balls - balls_bowled
        
        # Calculate required run rate for second innings
        required_run_rate = 0
        if self.current_innings == 2 and 'target' in innings_data:
            runs_needed = innings_data['target'] - innings_data['score']
            required_run_rate = (runs_needed * 6) / max(1, balls_remaining)
        
        # Create match state
        match_state = {
            'innings': self.current_innings,
            'is_batting': is_batting if is_batting is not None else True,
            'score': innings_data['score'],
            'wickets': innings_data['wickets'],
            'overs': innings_data['overs'],
            'balls': innings_data['balls'],
            'over': over,
            'current_phase': current_phase,
            'balls_remaining': balls_remaining,
            'wickets_remaining': 10 - innings_data['wickets'],
            'required_run_rate': required_run_rate
        }
        
        # Add additional state for bowling decisions
        if not match_state['is_batting']:
            match_state['last_bowler'] = innings_data['current_bowler']
            match_state['bowler_overs'] = {
                bowler_id: innings_data['bowler_stats'][bowler_id]['overs'] 
                for bowler_id in innings_data['bowler_stats']
            }
            
            # Add partnership info
            match_state['current_partnership_runs'] = innings_data['current_partnership']['runs']
            
            # Add recent boundary info
            recent_overs = min(3, innings_data['overs'])
            if recent_overs > 0:
                recent_boundaries = sum(
                    innings_data['over_history'].get(over, {}).get('boundary_count', 0)
                    for over in range(innings_data['overs']-recent_overs+1, innings_data['overs']+1)
                )
                match_state['recent_boundary_count'] = recent_boundaries
            
            # Add recent dot ball info
            if recent_overs > 0:
                recent_dots = sum(
                    innings_data['over_history'].get(over, {}).get('dot_count', 0)
                    for over in range(innings_data['overs']-recent_overs+1, innings_data['overs']+1)
                )
                match_state['recent_dot_count'] = recent_dots
        
        return match_state
    
    def _update_win_probability(self):
        """
        Calculate and update win probability based on current match state.
        """
        # Skip for first few balls when there's not enough data
        if self.innings_data[1]['balls'] < 6:
            return
        
        win_prob = 0.5  # Default to 50-50
        
        # If match is completed, probability is 0 or 1
        if self.match_completed:
            win_prob = 1.0 if self.winner == self.team1 else 0.0
        else:
            # Calculate based on current match state
            if self.current_innings == 1:
                # First innings - use projected score and historical data
                current_run_rate = self.innings_data[1]['score'] / max(1, self.innings_data[1]['overs'])
                projected_score = current_run_rate * 20
                
                # Adjust for wickets lost
                wicket_factor = max(0.7, 1.0 - (self.innings_data[1]['wickets'] * 0.05))
                projected_score *= wicket_factor
                
                # Compare to venue average
                venue_avg = self.venue_stats.get('first_innings_avg_score', 160)
                
                # Better than average score increases win probability
                if projected_score > venue_avg:
                    win_prob = 0.5 + min(0.4, (projected_score - venue_avg) / 100)
                else:
                    win_prob = 0.5 - min(0.4, (venue_avg - projected_score) / 100)
                
            else:
                # Second innings - use required run rate and wickets remaining
                target = self.innings_data[2]['target']
                current_score = self.innings_data[2]['score']
                balls_remaining = 120 - (self.innings_data[2]['overs'] * 6 + self.innings_data[2]['balls'] % 6)
                wickets_remaining = 10 - self.innings_data[2]['wickets']
                
                if balls_remaining == 0:
                    # Game over
                    if current_score >= target:
                        win_prob = 0.0  # Team 2 wins
                    else:
                        win_prob = 1.0  # Team 1 wins
                else:
                    # Calculate difficulty of chase
                    runs_needed = target - current_score
                    required_rr = (runs_needed * 6) / balls_remaining
                    
                    # Base win probability on required run rate and wickets
                    if required_rr <= 6:
                        # Easy chase favors batting team (team 2)
                        base_prob = 0.3
                    elif required_rr <= 8:
                        # Moderate chase
                        base_prob = 0.5
                    elif required_rr <= 10:
                        # Difficult chase
                        base_prob = 0.7
                    elif required_rr <= 12:
                        # Very difficult chase
                        base_prob = 0.8
                    else:
                        # Nearly impossible chase
                        base_prob = 0.9
                    
                    # Adjust for wickets remaining
                    wicket_adjustment = (10 - wickets_remaining) * 0.05
                    
                    # Final probability (from team 1's perspective)
                    win_prob = min(0.99, max(0.01, base_prob + wicket_adjustment))
        
        # Add to win probability tracking
        current_ball = self.innings_data[self.current_innings]['balls']
        current_over = self.innings_data[self.current_innings]['overs']
        
        self.win_probability.append({
            'innings': self.current_innings,
            'over': current_over,
            'ball': current_ball,
            'team1_win_prob': win_prob
        })
    
    def _check_for_key_moment(self, over, ball, innings_data, runs, is_wicket):
        """
        Identify and log key moments in the match.
        
        Args:
            over: Current over number
            ball: Current ball number
            innings_data: Current innings data
            runs: Runs scored on this ball
            is_wicket: Whether this ball resulted in a wicket
        """
        key_moment = None
        
        # Wicket is always a key moment
        if is_wicket:
            key_moment = {
                'type': 'wicket',
                'innings': self.current_innings,
                'over': over,
                'ball': ball,
                'batsman': innings_data['current_batsmen'][0],
                'bowler': innings_data['current_bowler'],
                'score': f"{innings_data['score']}/{innings_data['wickets']}"
            }
        
        # Boundary in a crucial phase
        elif runs >= 4:
            # Boundaries in death overs
            if over >= 16:
                key_moment = {
                    'type': 'death_boundary',
                    'innings': self.current_innings,
                    'over': over,
                    'ball': ball,
                    'runs': runs,
                    'batsman': innings_data['current_batsmen'][0],
                    'bowler': innings_data['current_bowler'],
                    'score': f"{innings_data['score']}/{innings_data['wickets']}"
                }
        
        # Big over (15+ runs)
        if ball == 6:  # End of over
            over_runs = innings_data['over_history'].get(over, {}).get('runs', 0)
            if over_runs >= 15:
                key_moment = {
                    'type': 'big_over',
                    'innings': self.current_innings,
                    'over': over,
                    'runs': over_runs,
                    'bowler': innings_data['current_bowler'],
                    'score': f"{innings_data['score']}/{innings_data['wickets']}"
                }
        
        # Momentum shift in second innings
        if self.current_innings == 2 and 'target' in innings_data:
            target = innings_data['target']
            current_score = innings_data['score']
            balls_remaining = 120 - ((over - 1) * 6 + ball)
            runs_needed = target - current_score
            
            # Team suddenly needing less than run a ball
            if runs_needed <= balls_remaining and runs >= 4:
                key_moment = {
                    'type': 'chase_momentum',
                    'innings': self.current_innings,
                    'over': over,
                    'ball': ball,
                    'runs_needed': runs_needed,
                    'balls_remaining': balls_remaining,
                    'score': f"{innings_data['score']}/{innings_data['wickets']}"
                }
        
        # Add key moment if identified
        if key_moment:
            self.key_moments.append(key_moment)
    
    def determine_winner(self):
        """
        Determine the winner of the match.
        """
        first_innings_score = self.innings_data[1]['score']
        second_innings_score = self.innings_data[2]['score']
        
        if second_innings_score > first_innings_score:
            self.winner = self.innings_data[2]['batting_team']
        elif first_innings_score > second_innings_score:
            self.winner = self.innings_data[1]['batting_team']
        else:
            # It's a tie
            self.winner = None
    
    def _get_victory_margin(self):
        """
        Calculate the margin of victory.
        
        Returns:
            String describing the victory margin
        """
        if not self.winner:
            return "Match tied"
        
        if self.winner == self.innings_data[1]['batting_team']:
            # Won by runs
            margin = self.innings_data[1]['score'] - self.innings_data[2]['score']
            return f"{margin} runs"
        else:
            # Won by wickets
            wickets_remaining = 10 - self.innings_data[2]['wickets']
            return f"{wickets_remaining} wickets"
    
    def get_match_results(self):
        """
        Generate comprehensive match results summary.
        
        Returns:
            Dictionary with detailed match results
        """
        results = {
            'match_id': id(self),
            'teams': {
                'team1': self.team1.name,
                'team2': self.team2.name
            },
            'toss': {
                'winner': self.toss_winner.name,
                'decision': self.toss_decision
            },
            'innings': {
                1: {
                    'batting_team': self.innings_data[1]['batting_team'].name,
                    'score': self.innings_data[1]['score'],
                    'wickets': self.innings_data[1]['wickets'],
                    'overs': f"{self.innings_data[1]['overs']}.{self.innings_data[1]['balls'] % 6}",
                    'run_rate': self.innings_data[1]['score'] / max(0.1, self.innings_data[1]['overs'])
                },
                2: {
                    'batting_team': self.innings_data[2]['batting_team'].name,
                    'score': self.innings_data[2]['score'],
                    'wickets': self.innings_data[2]['wickets'],
                    'overs': f"{self.innings_data[2]['overs']}.{self.innings_data[2]['balls'] % 6}",
                    'run_rate': self.innings_data[2]['score'] / max(0.1, self.innings_data[2]['overs'])
                }
            },
            'result': {
                'winner': self.winner.name if self.winner else "Tie",
                'margin': self._get_victory_margin()
            },
            'player_stats': {
                'batting': self._get_batting_stats(),
                'bowling': self._get_bowling_stats()
            }#,
            #'key_moments': self.key_moments,
            #'win_probability': self.win_probability
        }
        
        return results
    
    def _get_batting_stats(self):
        """
        Compile batting statistics from both innings.
        
        Returns:
            Dictionary with batting statistics
        """
        batting_stats = {}
        
        for innings in [1, 2]:
            team = self.innings_data[innings]['batting_team']
            
            for player_id, stats in self.innings_data[innings]['batsmen_stats'].items():
                if player_id in team.players:
                    player_name = team.players[player_id].name
                    
                    batting_stats[player_id] = {
                        'name': player_name,
                        'team': team.name,
                        'innings': innings,
                        'runs': stats['runs'],
                        'balls': stats['balls'],
                        'fours': stats['fours'],
                        'sixes': stats['sixes'],
                        'strike_rate': stats['strike_rate'],
                        'dismissed': player_id in self.innings_data[innings]['dismissed']
                    }
        
        return batting_stats
    
    def _get_bowling_stats(self):
        """
        Compile bowling statistics from both innings.
        
        Returns:
            Dictionary with bowling statistics
        """
        bowling_stats = {}
        
        for innings in [1, 2]:
            team = self.innings_data[innings]['bowling_team']
            
            for player_id, stats in self.innings_data[innings]['bowler_stats'].items():
                if player_id in team.players:
                    player_name = team.players[player_id].name
                    
                    bowling_stats[player_id] = {
                        'name': player_name,
                        'team': team.name,
                        'innings': innings,
                        'overs': stats['overs'],
                        'maidens': stats['maidens'],
                        'runs': stats['runs'],
                        'wickets': stats['wickets'],
                        'economy': stats['economy']
                    }
        
        return bowling_stats
    
    def __str__(self):
        """String representation of the match."""
        if not self.match_completed:
            return f"T20 Match: {self.team1.name} vs {self.team2.name} (Not started/In progress)"
        
        return f"T20 Match: {self.team1.name} vs {self.team2.name} - {self.winner.name if self.winner else 'Tie'} by {self._get_victory_margin()}"