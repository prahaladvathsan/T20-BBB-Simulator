"""
Advanced test for T20 Cricket Match Simulator components.
Simulates a 10-team league with 45 matches.
"""

import os
import sys
import json
import unittest
from datetime import datetime, timedelta
import random
import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class AdvancedSimulatorTest(unittest.TestCase):
    """
    Advanced tests for the T20 Cricket Match Simulator.
    Simulates a 10-team league with 45 matches.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create test data directory if it doesn't exist
        os.makedirs("tests/advanced_test_data", exist_ok=True)
        
        # Create basic test data files
        self._create_advanced_test_data()
        
        # Create initial BBB dataset
        self._create_initial_bbb_data() 

    def _create_advanced_test_data(self):
        """Create test data for 10 teams with enhanced player stats."""
        # Create squad profiles
        squad_data = {}
        player_counter = 1
        
        for team_num in range(1, 11):
            team_id = f"team{team_num}"
            players = []
            
            # Create 11 players for each team
            for _ in range(11):
                player_id = f"p{player_counter}"
                
                # Create players with only main_role, let Team class handle specific roles
                if player_counter % 11 in [1, 2, 3, 4, 5]:
                    role = "batsman"
                elif player_counter % 11 in [6, 7]:
                    role = "all-rounder"
                else:
                    role = "bowler"
                
                players.append({
                    "player_id": player_id,
                    "name": f"Player {player_counter}",
                    "main_role": role
                })
                
                player_counter += 1
            
            squad_data[team_id] = {
                "name": f"Team {team_num}",
                "players": players
            }
        
        # Create player stats
        self._create_player_stats(player_counter - 1)
        
        # Write data to files
        os.makedirs("tests/advanced_test_data/teams", exist_ok=True)
        os.makedirs("tests/advanced_test_data/players", exist_ok=True)
        os.makedirs("tests/advanced_test_data/matches", exist_ok=True)
        os.makedirs("tests/advanced_test_data/venues", exist_ok=True)
        
        with open("tests/advanced_test_data/teams/squad_profiles.json", "w") as f:
            json.dump(squad_data, f, indent=2)
        
        # Create sample venue stats
        self._create_venue_stats()
    
    def _create_player_stats(self, num_players):
        """Create batting and bowling stats for all players."""
        # Create simplified batting stats
        batting_stats = {}
        for i in range(1, num_players + 1):
            player_id = f"p{i}"
            
            # Add some variety based on player role
            role_mod = 1.0
            if i % 11 in [1, 2]:  # Openers
                role_mod = 1.5
            elif i % 11 in [3, 4, 5]:  # Middle order
                role_mod = 1.3
            elif i % 11 in [6, 7]:  # All-rounders
                role_mod = 1.0
            else:  # Bowlers
                role_mod = 0.5
                
            # Team-based variety
            team_mod = 0.9 + (((i - 1) // 11) * 0.02)
            
            # Base stats with modifications
            batting_stats[player_id] = {
                "bat_hand": "right" if i % 3 != 0 else "left", 
                "batter_id": player_id,
                "total_runs": int((500 + (i % 11 * 50)) * role_mod * team_mod),
                "total_balls": int((400 + (i % 11 * 20)) * team_mod),
                "dismissals": int(15 * team_mod),
                "strike_rate": ((500 + (i % 11 * 50)) * role_mod) / (400 + (i % 11 * 20)) * 100,
                "average": ((500 + (i % 11 * 50)) * role_mod) / (15 * team_mod),
                "by_phase": {
                    "1": {
                        "runs": int(150 * role_mod * team_mod), 
                        "balls": 100, 
                        "fours": int(8 * role_mod), 
                        "sixes": int(4 * role_mod), 
                        "dots": 50,
                        "dismissals": int(3 * team_mod),
                        "strike_rate": 150.0 * role_mod,
                        "average": 50.0 * role_mod
                    },
                    "2": {
                        "runs": int(150 * role_mod * team_mod), 
                        "balls": 100, 
                        "fours": int(7 * role_mod), 
                        "sixes": int(3 * role_mod), 
                        "dots": 50,
                        "dismissals": int(4 * team_mod),
                        "strike_rate": 150.0 * role_mod,
                        "average": 37.5 * role_mod
                    },
                    "3": {
                        "runs": int(100 * role_mod * team_mod), 
                        "balls": 100, 
                        "fours": int(5 * role_mod), 
                        "sixes": int(3 * role_mod), 
                        "dots": 50,
                        "dismissals": int(4 * team_mod),
                        "strike_rate": 100.0 * role_mod,
                        "average": 25.0 * role_mod
                    },
                    "4": {
                        "runs": int(100 * role_mod * team_mod), 
                        "balls": 100, 
                        "fours": int(5 * role_mod), 
                        "sixes": int(5 * role_mod), 
                        "dots": 50,
                        "dismissals": int(4 * team_mod),
                        "strike_rate": 100.0 * role_mod,
                        "average": 25.0 * role_mod
                    }
                },
                "by_position": {
                    str(pos): {
                        # Define bounds for each stat based on position
                        "runs": int(random.uniform(
                            (80 + (pos * 5)) * role_mod * team_mod,  # Lower bound
                            (120 + (pos * 15)) * role_mod * team_mod  # Upper bound
                        )),
                        "balls": int(random.uniform(
                            (60 + (pos * 3)) * team_mod,  # Lower bound
                            (100 + (pos * 8)) * team_mod  # Upper bound
                        )),
                        "dismissals": int(random.uniform(
                            2 * team_mod,  # Lower bound
                            4 * team_mod  # Upper bound
                        )),
                        "fours": int(random.uniform(
                            3 * role_mod,  # Lower bound
                            7 * role_mod  # Upper bound
                        )),
                        "sixes": int(random.uniform(
                            1 * role_mod,  # Lower bound
                            4 * role_mod  # Upper bound
                        )),
                        "dots": int(random.uniform(
                            15 * team_mod,  # Lower bound
                            25 * team_mod  # Upper bound
                        ))
                    } for pos in range(1, 12)
                },
                "vs_bowler_styles": {
                    "fast": {
                        "runs": int(250 * role_mod * team_mod), 
                        "balls": 200, 
                        "dismissals": int(8 * team_mod), 
                        "dots": 80,
                        "fours": int(15 * role_mod), 
                        "sixes": int(5 * role_mod)
                    },
                    "spin": {
                        "runs": int(250 * role_mod * team_mod), 
                        "balls": 200, 
                        "dismissals": int(7 * team_mod), 
                        "dots": 70,
                        "fours": int(10 * role_mod), 
                        "sixes": int(10 * role_mod)
                    }
                },
                "boundary_percentage": (20 + i % 11) / 100.0
            }

            # Calculate derived stats for by_position
            for pos in range(1, 12):
                pos_stats = batting_stats[player_id]["by_position"][str(pos)]
                runs = pos_stats["runs"]
                balls = pos_stats["balls"]
                dismissals = pos_stats["dismissals"]
                
                # Calculate strike rate and average
                pos_stats["strike_rate"] = round((runs / balls * 100) if balls > 0 else 0, 2)
                pos_stats["average"] = round(runs / dismissals if dismissals > 0 else runs, 2)

            # Calculate derived stats for vs_bowler_styles
            for style in ["fast", "spin"]:
                style_stats = batting_stats[player_id]["vs_bowler_styles"][style]
                runs = style_stats["runs"]
                balls = style_stats["balls"]
                dismissals = style_stats["dismissals"]
                
                # Calculate strike rate and average
                style_stats["strike_rate"] = round((runs / balls * 100) if balls > 0 else 0, 2)
                style_stats["average"] = round(runs / dismissals if dismissals > 0 else runs, 2)
        
        # Create simplified bowling stats
        bowling_stats = {}
        for i in range(1, num_players + 1):
            player_id = f"p{i}"
            
            # Add some variety based on player role
            role_mod = 1.0
            if i % 11 in [1, 2, 3, 4, 5]:  # Batsmen
                role_mod = 0.7
            elif i % 11 in [6, 7]:  # All-rounders
                role_mod = 1.0
            else:  # Bowlers
                role_mod = 1.2
                
            # Team-based variety
            team_mod = 0.9 + (((i - 1) // 11) * 0.02)
            
            bowling_stats[player_id] = {
                "bowling_style": "fast" if i % 3 != 0 else "spin",
                "bowler_id": player_id,
                "runs": int((400 + (i % 11 * 20)) * team_mod),
                "balls": int((300 + (i % 11 * 10)) * role_mod * team_mod),
                "wickets": int((15 + (i % 5)) * role_mod * team_mod),
                "economy": ((400 + (i % 11 * 20)) / (300 + (i % 11 * 10) * role_mod)) * 6,
                "by_phase": {
                    "1": {
                        "runs": int(120 * team_mod), 
                        "balls": int(80 * role_mod * team_mod), 
                        "wickets": int(3 * role_mod * team_mod), 
                        "dots": int(40 * role_mod),
                        "fours": 8, 
                        "sixes": 2
                    },
                    "2": {
                        "runs": int(100 * team_mod), 
                        "balls": int(80 * role_mod * team_mod), 
                        "wickets": int(4 * role_mod * team_mod), 
                        "dots": int(35 * role_mod),
                        "fours": 5, 
                        "sixes": 1
                    },
                    "3": {
                        "runs": int(90 * team_mod), 
                        "balls": int(70 * role_mod * team_mod), 
                        "wickets": int(4 * role_mod * team_mod), 
                        "dots": int(30 * role_mod),
                        "fours": 5, 
                        "sixes": 2
                    },
                    "4": {
                        "runs": int(90 * team_mod), 
                        "balls": int(70 * role_mod * team_mod), 
                        "wickets": int(4 * role_mod * team_mod), 
                        "dots": int(25 * role_mod),
                        "fours": 4, 
                        "sixes": 4
                    }
                },
                "by_over": {
                    str(over): {
                        # Define bounds for each stat based on over number
                        "runs": int(random.uniform(
                            5 * team_mod,  # Lower bound
                            (10 + (over % 3)) * team_mod  # Upper bound
                        )),
                        "balls": 6,  # Always 6 balls in an over
                        "wickets": int(random.uniform(
                            0.1 * role_mod * team_mod,  # Lower bound
                            0.5 * role_mod * team_mod  # Upper bound
                        )),
                        "dots": int(random.uniform(
                            1 * role_mod,  # Lower bound
                            3 * role_mod  # Upper bound
                        )),
                        "fours": int(random.uniform(
                            0.2,  # Lower bound
                            1.0  # Upper bound
                        )),
                        "sixes": int(random.uniform(
                            0.1,  # Lower bound
                            0.5  # Upper bound
                        ))
                    } for over in range(1, 21)
                },
                "vs_bat_hand": {
                    "rhb": {
                        "runs": int(200 * team_mod), 
                        "balls": int(150 * role_mod * team_mod), 
                        "wickets": int(8 * role_mod * team_mod),
                        "dots": int(50 * role_mod),
                        "fours": 12, 
                        "sixes": 7
                    },
                    "lhb": {
                        "runs": int(200 * team_mod), 
                        "balls": int(150 * role_mod * team_mod), 
                        "wickets": int(7 * role_mod * team_mod),
                        "dots": int(60 * role_mod),
                        "fours": 8, 
                        "sixes": 4
                    }
                }
            }

            # Calculate derived stats for by_over
            for over in range(1, 21):
                over_stats = bowling_stats[player_id]["by_over"][str(over)]
                runs = over_stats["runs"]
                balls = over_stats["balls"]
                wickets = over_stats["wickets"]
                
                # Calculate economy, average, and bowling strike rate
                over_stats["economy"] = round((runs / (balls/6)) if balls > 0 else 0, 2)
                over_stats["average"] = round(runs / wickets if wickets > 0 else runs, 2)
                over_stats["bowling_sr"] = round(balls / wickets if wickets > 0 else balls, 2)

            # Calculate derived stats for by_phase
            for phase in ["1", "2", "3", "4"]:
                phase_stats = bowling_stats[player_id]["by_phase"][phase]
                runs = phase_stats["runs"]
                balls = phase_stats["balls"]
                wickets = phase_stats["wickets"]
                
                # Calculate economy, average, and bowling strike rate
                phase_stats["economy"] = round((runs / (balls/6)) if balls > 0 else 0, 2)
                phase_stats["average"] = round(runs / wickets if wickets > 0 else runs, 2)
                phase_stats["bowling_sr"] = round(balls / wickets if wickets > 0 else balls, 2)

            # Calculate derived stats for vs_bat_hand
            for hand in ["rhb", "lhb"]:
                hand_stats = bowling_stats[player_id]["vs_bat_hand"][hand]
                runs = hand_stats["runs"]
                balls = hand_stats["balls"]
                wickets = hand_stats["wickets"]
                
                # Calculate economy, average, and bowling strike rate
                hand_stats["economy"] = round((runs / (balls/6)) if balls > 0 else 0, 2)
                hand_stats["average"] = round(runs / wickets if wickets > 0 else runs, 2)
                hand_stats["bowling_sr"] = round(balls / wickets if wickets > 0 else balls, 2)
        
        # Write data to files
        with open("tests/advanced_test_data/players/batting_stats.json", "w") as f:
            json.dump(batting_stats, f, indent=2)
        
        with open("tests/advanced_test_data/players/bowling_stats.json", "w") as f:
            json.dump(bowling_stats, f, indent=2)
            
    def _create_venue_stats(self):
        """Create sample venue stats for testing."""
        venue_stats = {
            "venue1": {
                "first_innings_avg_score": 165,
                "first_innings_std_score": 20,
                "second_innings_avg_score": 155,
                "second_innings_std_score": 25,
                "matches_played": 20,
                "phase_1_run_rate": 8.2,  # Powerplay
                "phase_2_run_rate": 7.5,  # Early Middle
                "phase_3_run_rate": 8.8,  # Late Middle
                "phase_4_run_rate": 10.1, # Death
                "phase_1_wicket_rate": 0.8,
                "phase_2_wicket_rate": 1.2,
                "phase_3_wicket_rate": 1.4,
                "phase_4_wicket_rate": 1.6
            },
            "venue2": {
                "first_innings_avg_score": 175,
                "first_innings_std_score": 18,
                "second_innings_avg_score": 165,
                "second_innings_std_score": 22,
                "matches_played": 15,
                "phase_1_run_rate": 8.5,  # Powerplay
                "phase_2_run_rate": 7.8,  # Early Middle
                "phase_3_run_rate": 9.2,  # Late Middle
                "phase_4_run_rate": 10.5, # Death
                "phase_1_wicket_rate": 0.7,
                "phase_2_wicket_rate": 1.1,
                "phase_3_wicket_rate": 1.3,
                "phase_4_wicket_rate": 1.5
            },
            "venue3": {
                "first_innings_avg_score": 155,
                "first_innings_std_score": 22,
                "second_innings_avg_score": 145,
                "second_innings_std_score": 28,
                "matches_played": 25,
                "phase_1_run_rate": 7.8,  # Powerplay
                "phase_2_run_rate": 7.2,  # Early Middle
                "phase_3_run_rate": 8.5,  # Late Middle
                "phase_4_run_rate": 9.7,  # Death
                "phase_1_wicket_rate": 0.9,
                "phase_2_wicket_rate": 1.3,
                "phase_3_wicket_rate": 1.5,
                "phase_4_wicket_rate": 1.7
            }
        }
        
        with open("tests/advanced_test_data/venues/venue_stats.json", "w") as f:
            json.dump(venue_stats, f, indent=2) 

    def _create_initial_bbb_data(self):
        """Create the initial BBB dataset structure."""
        os.makedirs("tests/advanced_test_data/matches", exist_ok=True)
        
        # Define the BBB dataset structure
        fieldnames = [
            'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
            'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
            'runs', 'is_wicket', 'outcome', 'line', 'length', 'phase'
        ]
        
        # Create empty CSV file with headers
        csv_path = "tests/advanced_test_data/matches/t20_bbb.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def _create_match_schedule(self):
        """Create a round-robin tournament schedule for 10 teams (45 matches)."""
        teams = [f"team{i}" for i in range(1, 11)]
        matches = []
        
        # Round-robin scheduling algorithm (each team plays against every other team)
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                matches.append((teams[i], teams[j]))
        
        # Randomize the order of matches
        random.shuffle(matches)
        
        return matches
    
    def _simulate_league(self):
        """Simulate all matches in the league and record results."""
        from src.utils.data_loader import DataLoader
        from src.models.player import Player
        from src.models.team import Team
        from src.models.match import Match
        
        # Load test data
        loader = DataLoader("tests/advanced_test_data")
        linked_data = loader.load_comprehensive_data()
        
        # Create player objects
        player_objects = {}
        for player_id, player_data in linked_data['players'].items():
            player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        team_objects = {}
        for team_id, team_data in linked_data['teams'].items():
            team_objects[team_id] = Team(team_id, team_data, player_objects)
        
        # Generate match schedule
        matches = self._create_match_schedule()
        
        # Results storage
        league_results = []
        
        for match_num, (team1_id, team2_id) in enumerate(matches, 1):
            print(f"Simulating match {match_num}: {team1_id} vs {team2_id}")
            
            team1 = team_objects[team1_id]
            team2 = team_objects[team2_id]
            
            # Rotate between 3 venues
            venue_id = f"venue{(match_num % 3) + 1}"
            
            # Match date (simulating a league over 45 days)
            match_date = (datetime.now() - timedelta(days=45-match_num)).strftime('%Y-%m-%d')
            
            # Create a match with a unique ID
            match = Match(team1, team2, match_id=f"match{match_num}", venue_id=venue_id, match_date=match_date)
            
            # Simulate the match
            match_results = match.simulate_match()

            # Display the match scorecard
            self.display_match_scorecard(match_results)
            
            # Get BBB data directly from match results and append to dataset
            bbb_data = match_results.get('bbb_data', [])
            self._append_to_bbb_dataset(bbb_data)
            
            # Store match results for standings calculation
            winner_name = match_results['result']['winner']
            winner_id = team1_id if winner_name == team1.name else team2_id
            
            league_results.append({
                'match_num': match_num,
                'team1': team1_id,
                'team2': team2_id,
                'winner': winner_id,
                'margin': match_results['result']['margin'],
                'margin_type': match_results.get('margin_type', 'runs' if match_results['innings'][1]['batting_team'] == winner_name else 'wickets')
            })
        
        return league_results
        
    def _append_to_bbb_dataset(self, bbb_data):
        """Append new ball-by-ball data to the dataset."""
        if not bbb_data:
            return
            
        csv_path = "tests/advanced_test_data/matches/t20_bbb.csv"
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = [
                'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
                'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
                'runs', 'is_wicket', 'outcome', 'line', 'length', 'phase'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(bbb_data) 

    def _calculate_standings(self, league_results):
        """Calculate league standings from match results."""
        # Initialize standings
        standings = {}
        
        for team_id in [f"team{i}" for i in range(1, 11)]:
            standings[team_id] = {
                'played': 0, 'won': 0, 'lost': 0, 
                'points': 0, 'nrr': 0.0
            }
        
        for match in league_results:
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
    
    def _display_league_table(self, standings):
        """Display league standings in a formatted table."""
        print("\n" + "="*80)
        print(f"{'LEAGUE STANDINGS':^80}")
        print("="*80)
        
        print(f"{'TEAM':<20} {'P':<5} {'W':<5} {'L':<5} {'PTS':<5} {'NRR':<8}")
        print("-"*80)
        
        for team, stats in standings:
            print(f"{team:<20} {stats['played']:<5} {stats['won']:<5} {stats['lost']:<5} {stats['points']:<5} {stats['nrr']:<8.3f}")
        
        print("="*80)
    
    def test_league_simulation(self):
        """Test simulating a 10-team league with 45 matches."""
        # Simulate the league
        league_results = self._simulate_league()
        
        # Calculate standings
        standings = self._calculate_standings(league_results)
        
        # Display league table
        self._display_league_table(standings)
        
        # Verify the simulation worked correctly
        self.assertEqual(len(league_results), 45, "Should have simulated 45 matches")
        
        # Check that BBB data was generated
        try:
            bbb_data = pd.read_csv("tests/advanced_test_data/matches/t20_bbb.csv")
            self.assertGreater(len(bbb_data), 0, "BBB data should have been generated")
            print(f"\nGenerated {len(bbb_data)} ball-by-ball records across 45 matches")
            
            # Check the number of matches in the BBB data
            unique_matches = bbb_data['match_id'].unique()
            self.assertEqual(len(unique_matches), 45, "BBB data should contain records for all 45 matches")
            
            # Check that required columns exist
            required_columns = [
                'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
                'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
                'runs', 'is_wicket', 'outcome', 'phase'
            ]
            for col in required_columns:
                self.assertIn(col, bbb_data.columns, f"Column {col} should exist in BBB data")
                
            # Print some stats about the BBB data
            print("\nBall-by-ball data statistics:")
            print(f"Total runs scored: {bbb_data['runs'].sum()}")
            print(f"Total wickets: {bbb_data['is_wicket'].sum()}")
            print(f"Average runs per match: {bbb_data.groupby('match_id')['runs'].sum().mean():.2f}")
            
        except Exception as e:
            self.fail(f"Error checking BBB data: {str(e)}")
            
        print("\nAdvanced test completed successfully!")

    def display_match_scorecard(self, match_results):
        """Display match results in a nicely formatted cricket scorecard."""
        """Display match results in a nicely formatted cricket scorecard."""
        print("\n" + "="*80)
        print(f"{'TEST MATCH SCORECARD':^80}")
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
                print(f"\nInnings {innings_num} data not found")
                continue
                
            innings = match_results['innings'][innings_num]
            batting_team = innings.get('batting_team', f"Team {innings_num}")
            score = innings.get('score', 0)
            wickets = innings.get('wickets', 0)
            overs = innings.get('overs', 20)
            
            # Format overs display
            overs_str = str(overs)
            if isinstance(overs, str) and '.' in overs:
                # Already in the right format
                pass
            elif isinstance(overs, (int, float)):
                # Convert to proper cricket notation (e.g., 20.5 is 20 overs and 5 balls)
                if isinstance(overs, float):
                    overs_int = int(overs)
                    balls = int(round((overs - overs_int) * 6))
                    overs_str = f"{overs_int}.{balls}"
            
            print(f"\n{batting_team} INNINGS: {score}/{wickets} ({overs_str} overs)")
            print("-"*80)
            
            # Display batsmen performances
            print(f"{'BATSMAN':<25} {'STATUS':<15} {'RUNS':<8} {'BALLS':<8} {'4s':<5} {'6s':<5} {'SR':<8}")
            print("-"*80)
            
            # Get batting performances for this innings
            batsmen = []
            if 'player_stats' in match_results and 'batting' in match_results['player_stats']:
                for player_id, stats in match_results['player_stats']['batting'].items():
                    # Check if this player batted in this innings
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
            
            if not batsmen:
                print("No batting data available")
            else:
                # Sort by batting position
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
                # Determine bowling team from match_results
                team1 = match_results['teams'].get('team1', '')
                team2 = match_results['teams'].get('team2', '')
                bowling_team = team2 if batting_team == team1 else team1
                    
            print(f"\n{bowling_team} BOWLING:")
            print(f"{'BOWLER':<25} {'O':<5} {'M':<5} {'R':<5} {'W':<5} {'ECON':<8}")
            print("-"*80)
            
            # Get bowling performances for this innings
            bowlers = []
            if 'player_stats' in match_results and 'bowling' in match_results['player_stats']:
                for player_id, stats in match_results['player_stats']['bowling'].items():
                    # Check if this player bowled in this innings
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
            
            if not bowlers:
                print("No bowling data available")
            else:
                # Sort by number of overs bowled (most first)
                bowlers.sort(key=lambda x: (x.get('overs', 0), x.get('balls', 0)), reverse=True)
                
                for bowler in bowlers:
                    name = bowler.get('name', bowler['player_id'])
                    
                    # Get bowling stats
                    overs = bowler.get('overs', 0)
                    balls = bowler.get('balls', 0)
                    
                    # Format overs (convert balls to overs if needed)
                    if isinstance(overs, (int, float)) and overs > 0:
                        # Convert to proper cricket notation
                        if isinstance(overs, float):
                            overs_int = int(overs)
                            balls_part = int(round((overs - overs_int) * 6))
                            overs_str = f"{overs_int}.{balls_part}"
                        else:
                            overs_str = f"{overs}.0"
                    else:
                        # Calculate from balls
                        overs_int = balls // 6
                        remainder = balls % 6
                        overs_str = f"{overs_int}.{remainder}"
                        
                    runs = bowler.get('runs', 0)
                    wickets = bowler.get('wickets', 0)
                    maidens = bowler.get('maidens', 0)
                    
                    # Calculate economy rate
                    if isinstance(overs, (int, float)) and overs > 0:
                        econ = round(runs / overs, 2)
                    else:
                        # Calculate from balls
                        econ = round((runs / (balls/6)) if balls > 0 else 0, 2)
                    
                    print(f"{name:<25} {overs_str:<5} {maidens:<5} {runs:<5} {wickets:<5} {econ:<8}")
            
            print("="*80)
        


if __name__ == "__main__":
    unittest.main() 