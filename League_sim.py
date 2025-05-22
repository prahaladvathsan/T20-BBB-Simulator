"""
League Simulator for T20 Cricket Matches using real data.
This script simulates a complete league using actual team and player data.
"""

import os
import sys
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils.data_loader import DataLoader
from src.models.player import Player
from src.models.team import Team
from src.models.match import Match

class LeagueSimulator:
    """
    Simulates a complete T20 cricket league using real team and player data.
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the league simulator with data directory."""
        self.data_dir = data_dir
        self.loader = DataLoader(data_dir)
        self.linked_data = None
        self.player_objects = {}
        self.team_objects = {}
        self.league_results = []
        
    def load_data(self):
        """Load all required data for simulation."""
        print("Loading data...")
        self.linked_data = self.loader.load_comprehensive_data()
        
        # Create player objects
        for player_id, player_data in self.linked_data['players'].items():
            self.player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        for team_id, team_data in self.linked_data['teams'].items():
            print(team_id, team_data)
            self.team_objects[team_id] = Team(team_id, team_data, self.player_objects)
            
        print(f"Loaded data for {len(self.team_objects)} teams and {len(self.player_objects)} players")
        
    def create_match_schedule(self) -> List[Tuple[str, str]]:
        """Create a round-robin tournament schedule."""
        teams = list(self.team_objects.keys())
        matches = []
        
        # Round-robin scheduling (each team plays against every other team)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matches.append((teams[i], teams[j]))
        
        # Randomize the order of matches
        random.shuffle(matches)
        return matches
    
    def simulate_league(self):
        """Simulate all matches in the league and record results."""
        if not self.linked_data:
            self.load_data()
            
        matches = self.create_match_schedule()
        total_matches = len(matches)
        
        print(f"\nStarting league simulation with {total_matches} matches...")
        
        # Create output directory for BBB data if it doesn't exist
        os.makedirs("output/matches", exist_ok=True)
        
        # Initialize BBB dataset
        self._initialize_bbb_dataset()
        
        for match_num, (team1_id, team2_id) in enumerate(matches, 1):
            print(f"\nSimulating match {match_num}/{total_matches}: {team1_id} vs {team2_id}")
            
            team1 = self.team_objects[team1_id]
            team2 = self.team_objects[team2_id]
            
            # Get venue data
            venues = list(self.linked_data['venues'].keys())
            venue_id = random.choice(venues)
            
            # Match date (simulating a league over the number of matches)
            match_date = (datetime.now() - timedelta(days=total_matches-match_num)).strftime('%Y-%m-%d')
            
            # Create and simulate match
            match = Match(team1, team2, match_id=f"match{match_num}", venue_id=venue_id, match_date=match_date)
            match_results = match.simulate_match()
            
            # Display match scorecard
            self.display_match_scorecard(match_results)
            
            # Get BBB data and append to dataset
            bbb_data = match_results.get('bbb_data', [])
            self._append_to_bbb_dataset(bbb_data)
            
            # Store match results
            winner_name = match_results['result']['winner']
            winner_id = team1_id if winner_name == team1.name else team2_id
            
            self.league_results.append({
                'match_num': match_num,
                'team1': team1_id,
                'team2': team2_id,
                'winner': winner_id,
                'margin': match_results['result']['margin'],
                'margin_type': match_results.get('margin_type', 'runs' if match_results['innings'][1]['batting_team'] == winner_name else 'wickets')
            })
            
        return self.league_results
    
    def _initialize_bbb_dataset(self):
        """Initialize the ball-by-ball dataset structure."""
        fieldnames = [
            'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
            'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
            'runs', 'is_wicket', 'outcome', 'line', 'length', 'phase'
        ]
        
        csv_path = "output/matches/t20_bbb.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def _append_to_bbb_dataset(self, bbb_data: List[Dict]):
        """Append new ball-by-ball data to the dataset."""
        if not bbb_data:
            return
            
        csv_path = "output/matches/t20_bbb.csv"
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
                'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
                'runs', 'is_wicket', 'outcome', 'line', 'length', 'phase'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(bbb_data)
    
    def calculate_standings(self) -> List[Tuple[str, Dict]]:
        """Calculate league standings from match results."""
        standings = {}
        
        for team_id in self.team_objects.keys():
            standings[team_id] = {
                'played': 0, 'won': 0, 'lost': 0, 
                'points': 0, 'nrr': 0.0
            }
        
        for match in self.league_results:
            team1 = match['team1']
            team2 = match['team2']
            winner = match['winner']
            
            # Update match count
            standings[team1]['played'] += 1
            standings[team2]['played'] += 1
            
            # Update win/loss
            if winner == team1:
                standings[team1]['won'] += 1
                standings[team1]['points'] += 2
                standings[team2]['lost'] += 1
            else:
                standings[team2]['won'] += 1
                standings[team2]['points'] += 2
                standings[team1]['lost'] += 1
                
        # Sort by points, then NRR
        sorted_standings = sorted(
            standings.items(),
            key=lambda x: (x[1]['points'], x[1]['nrr']),
            reverse=True
        )
        
        return sorted_standings
    
    def display_league_table(self, standings: List[Tuple[str, Dict]]):
        """Display league standings in a formatted table."""
        print("\n" + "="*80)
        print(f"{'LEAGUE STANDINGS':^80}")
        print("="*80)
        
        print(f"{'TEAM':<20} {'P':<5} {'W':<5} {'L':<5} {'PTS':<5} {'NRR':<8}")
        print("-"*80)
        
        for team, stats in standings:
            print(f"{team:<20} {stats['played']:<5} {stats['won']:<5} {stats['lost']:<5} {stats['points']:<5} {stats['nrr']:<8.3f}")
        
        print("="*80)
    
    def display_match_scorecard(self, match_results: Dict):
        """Display match results in a nicely formatted cricket scorecard."""
        print("\n" + "="*80)
        print(f"{'MATCH SCORECARD':^80}")
        print("="*80)
        
        # Match result summary
        winner = match_results['result'].get('winner', 'Unknown')
        margin = match_results['result'].get('margin', 0)
        margin_type = match_results['result'].get('margin_type', 'runs')
        
        print(f"\nRESULT: {winner} won by {margin} {margin_type}")
        print("-"*80)
        
        # Display both innings
        for innings_num in [1, 2]:
            if innings_num not in match_results['innings']:
                continue
                
            innings = match_results['innings'][innings_num]
            batting_team = innings.get('batting_team', f"Team {innings_num}")
            score = innings.get('score', 0)
            wickets = innings.get('wickets', 0)
            overs = innings.get('overs', 20)
            
            # Format overs display
            overs_str = str(overs)
            if isinstance(overs, float):
                overs_int = int(overs)
                balls = int(round((overs - overs_int) * 6))
                overs_str = f"{overs_int}.{balls}"
            
            print(f"\n{batting_team} INNINGS: {score}/{wickets} ({overs_str} overs)")
            print("-"*80)
            
            # Display batsmen performances
            print(f"{'BATSMAN':<25} {'STATUS':<15} {'RUNS':<8} {'BALLS':<8} {'4s':<5} {'6s':<5} {'SR':<8}")
            print("-"*80)
            
            # Get batting performances
            batsmen = []
            if 'player_stats' in match_results and 'batting' in match_results['player_stats']:
                for player_id, stats in match_results['player_stats']['batting'].items():
                    if stats.get('innings') == innings_num:
                        batsmen.append({
                            'player_id': player_id,
                            'name': stats.get('name', player_id),
                            'runs': stats.get('runs', 0),
                            'balls': stats.get('balls', 0),
                            'fours': stats.get('fours', 0),
                            'sixes': stats.get('sixes', 0),
                            'dismissed': stats.get('dismissed', False),
                            'batting_position': stats.get('position', 999)
                        })
            
            if batsmen:
                batsmen.sort(key=lambda x: x.get('batting_position', 999))
                for batsman in batsmen:
                    name = batsman.get('name', batsman['player_id'])
                    status = "not out" if not batsman.get('dismissed', False) else "out"
                    runs = batsman.get('runs', 0)
                    balls = batsman.get('balls', 0)
                    fours = batsman.get('fours', 0)
                    sixes = batsman.get('sixes', 0)
                    sr = round((runs / balls * 100) if balls > 0 else 0, 2)
                    
                    print(f"{name:<25} {status:<15} {runs:<8} {balls:<8} {fours:<5} {sixes:<5} {sr:<8}")
            
            print("-"*80)
            
            # Display extras and total
            extras = innings.get('extras', 0)
            wides = innings.get('wides', 0)
            no_balls = innings.get('no_balls', 0) 
            byes = innings.get('byes', 0)
            leg_byes = innings.get('leg_byes', 0)
            print(f"Extras: {extras} (WD: {wides}, NB: {no_balls}, B: {byes}, LB: {leg_byes})")
            print(f"TOTAL: {score}/{wickets} in {overs_str} overs")
            print("-"*80)
            
            # Display bowling performances
            bowling_team = innings.get('bowling_team', '')
            if not bowling_team:
                team1 = match_results['teams'].get('team1', '')
                team2 = match_results['teams'].get('team2', '')
                bowling_team = team2 if batting_team == team1 else team1
                    
            print(f"\n{bowling_team} BOWLING:")
            print(f"{'BOWLER':<25} {'O':<5} {'M':<5} {'R':<5} {'W':<5} {'ECON':<8}")
            print("-"*80)
            
            # Get bowling performances
            bowlers = []
            if 'player_stats' in match_results and 'bowling' in match_results['player_stats']:
                for player_id, stats in match_results['player_stats']['bowling'].items():
                    if stats.get('innings') == innings_num:
                        bowlers.append({
                            'player_id': player_id,
                            'name': stats.get('name', player_id),
                            'overs': stats.get('overs', 0),
                            'balls': stats.get('balls', 0),
                            'runs': stats.get('runs', 0),
                            'wickets': stats.get('wickets', 0),
                            'maidens': stats.get('maidens', 0)
                        })
            
            if bowlers:
                bowlers.sort(key=lambda x: (x.get('overs', 0), x.get('balls', 0)), reverse=True)
                for bowler in bowlers:
                    name = bowler.get('name', bowler['player_id'])
                    overs = bowler.get('overs', 0)
                    balls = bowler.get('balls', 0)
                    
                    if isinstance(overs, float):
                        overs_int = int(overs)
                        balls_part = int(round((overs - overs_int) * 6))
                        overs_str = f"{overs_int}.{balls_part}"
                    else:
                        overs_str = f"{overs}.0"
                        
                    runs = bowler.get('runs', 0)
                    wickets = bowler.get('wickets', 0)
                    maidens = bowler.get('maidens', 0)
                    econ = round(runs / overs if overs > 0 else 0, 2)
                    
                    print(f"{name:<25} {overs_str:<5} {maidens:<5} {runs:<5} {wickets:<5} {econ:<8}")
            
            print("="*80)

def main():
    """Main function to run the league simulation."""
    # Create simulator instance
    simulator = LeagueSimulator()
    
    # Simulate the league
    simulator.simulate_league()
    
    # Calculate and display standings
    standings = simulator.calculate_standings()
    simulator.display_league_table(standings)
    
    # Print summary of generated BBB data
    try:
        bbb_data = pd.read_csv("output/matches/t20_bbb.csv")
        print(f"\nGenerated {len(bbb_data)} ball-by-ball records")
        print(f"Total runs scored: {bbb_data['runs'].sum()}")
        print(f"Total wickets: {bbb_data['is_wicket'].sum()}")
        print(f"Average runs per match: {bbb_data.groupby('match_id')['runs'].sum().mean():.2f}")
    except Exception as e:
        print(f"Error reading BBB data: {str(e)}")

if __name__ == "__main__":
    main() 