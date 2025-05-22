"""
Basic test for T20 Cricket Match Simulator components.
Run this to verify your implementation works correctly.
"""

import os
import sys
import json
import unittest
from datetime import datetime
import random

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BasicSimulatorTest(unittest.TestCase):
    """
    Basic tests for the T20 Cricket Match Simulator components.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Create test data directory if it doesn't exist
        os.makedirs("tests/test_data", exist_ok=True)
        
        # Create basic test data files
        self._create_test_data()
    
    def _create_test_data(self):
        """Create minimal test data files."""
        # Create squad profiles
        squad_data = {
            "team1": {
                "name": "Test Team 1",
                "players": [
                    {"player_id": "p1", "name": "Player 1", "main_role": "batsman", "specific_roles": ["opener"]},
                    {"player_id": "p2", "name": "Player 2", "main_role": "batsman", "specific_roles": ["opener"]},
                    {"player_id": "p3", "name": "Player 3", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p4", "name": "Player 4", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p5", "name": "Player 5", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p6", "name": "Player 6", "main_role": "all-rounder", "specific_roles": ["finisher", "middle-overs-specialist"]},
                    {"player_id": "p7", "name": "Player 7", "main_role": "all-rounder", "specific_roles": ["finisher", "death-specialist"]},
                    {"player_id": "p8", "name": "Player 8", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]},
                    {"player_id": "p9", "name": "Player 9", "main_role": "bowler", "specific_roles": ["middle-overs-specialist"]},
                    {"player_id": "p10", "name": "Player 10", "main_role": "bowler", "specific_roles": ["death-specialist"]},
                    {"player_id": "p11", "name": "Player 11", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]}
                ]
            },
            "team2": {
                "name": "Test Team 2",
                "players": [
                    {"player_id": "p12", "name": "Player 12", "main_role": "batsman", "specific_roles": ["opener"]},
                    {"player_id": "p13", "name": "Player 13", "main_role": "batsman", "specific_roles": ["opener"]},
                    {"player_id": "p14", "name": "Player 14", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p15", "name": "Player 15", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p16", "name": "Player 16", "main_role": "batsman", "specific_roles": ["middle-order"]},
                    {"player_id": "p17", "name": "Player 17", "main_role": "all-rounder", "specific_roles": ["finisher", "middle-overs-specialist"]},
                    {"player_id": "p18", "name": "Player 18", "main_role": "all-rounder", "specific_roles": ["finisher", "death-specialist"]},
                    {"player_id": "p19", "name": "Player 19", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]},
                    {"player_id": "p20", "name": "Player 20", "main_role": "bowler", "specific_roles": ["middle-overs-specialist"]},
                    {"player_id": "p21", "name": "Player 21", "main_role": "bowler", "specific_roles": ["death-specialist"]},
                    {"player_id": "p22", "name": "Player 22", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]}
                ]
            }
        }
        
        # Create simplified batting stats
        batting_stats = {}
        for i in range(1, 23):  # Extended to 22 players (11 per team)
            player_id = f"p{i}"
            batting_stats[player_id] = {
                "bat_hand": "right", 
                "batter_id": player_id,
                "total_runs": 500 + (i * 50),
                "total_balls": 400 + (i * 20),
                "dismissals": 15,
                "strike_rate": (500 + (i * 50)) / (400 + (i * 20)) * 100,
                "average": (500 + (i * 50)) / 15,
                "by_phase": {
                    "1": {
                        "runs": 150, "balls": 100, "fours": 8, "sixes": 4, "dots": 50, "dismissals": 3,
                        "strike_rate": 150.0, "average": 50.0
                    },  # Powerplay (1-6)
                    "2": {
                        "runs": 150, "balls": 100, "fours": 7, "sixes": 3, "dots": 50, "dismissals": 4,
                        "strike_rate": 150.0, "average": 37.5
                    },  # Early Middle (7-12)
                    "3": {
                        "runs": 100, "balls": 100, "fours": 5, "sixes": 3, "dots": 50, "dismissals": 4,
                        "strike_rate": 100.0, "average": 25.0
                    },  # Late Middle (13-16)
                    "4": {
                        "runs": 100, "balls": 100, "fours": 5, "sixes": 5, "dots": 50, "dismissals": 4,
                        "strike_rate": 100.0, "average": 25.0
                    }   # Death (17-20)
                },
                "vs_bowler_styles": {
                    "fast": {
                        "runs": 250, "balls": 200, "dismissals": 8, "dots": 80,
                        "fours": 15, "sixes": 5
                    },
                    "spin": {
                        "runs": 250, "balls": 200, "dismissals": 7, "dots": 70,
                        "fours": 10, "sixes": 10
                    }
                },
                "boundary_percentage": (20 + i) / 100.0  # For testing lower_order_score calculation
            }
        
        # Create simplified bowling stats
        bowling_stats = {}
        for i in range(1, 23):  # Extended to 22 players (11 per team)
            player_id = f"p{i}"
            bowling_stats[player_id] = {
                "bowling_style": "fast" if i % 2 == 0 else "spin",
                "bowler_id": player_id,
                "runs": 400 + (i * 20),
                "balls": 300 + (i * 10),
                "wickets": 15 + (i % 5),
                "economy": ((400 + (i * 20)) / (300 + (i * 10))) * 6,
                "by_phase": {
                    "1": {
                        "runs": 120, "balls": 80, "wickets": 3, "dots": 40,
                        "fours": 8, "sixes": 2
                    },  # Powerplay
                    "2": {
                        "runs": 100, "balls": 80, "wickets": 4, "dots": 35,
                        "fours": 5, "sixes": 1
                    },  # Early Middle
                    "3": {
                        "runs": 90, "balls": 70, "wickets": 4, "dots": 30,
                        "fours": 5, "sixes": 2
                    },   # Late Middle  
                    "4": {
                        "runs": 90, "balls": 70, "wickets": 4, "dots": 25,
                        "fours": 4, "sixes": 4
                    }    # Death
                },
                "vs_batsman_types": {
                    "aggressive": {
                        "runs": 200, "balls": 150, "wickets": 8,
                        "dots": 50, "fours": 12, "sixes": 7
                    },
                    "anchor": {
                        "runs": 200, "balls": 150, "wickets": 7,
                        "dots": 60, "fours": 8, "sixes": 4
                    }
                }
            }
        
        # Write data to files
        os.makedirs("tests/test_data/teams", exist_ok=True)
        os.makedirs("tests/test_data/players", exist_ok=True)
        os.makedirs("tests/test_data/matches", exist_ok=True)
        os.makedirs("tests/test_data/venues", exist_ok=True)
        
        with open("tests/test_data/teams/squad_profiles.json", "w") as f:
            json.dump(squad_data, f, indent=2)
        
        with open("tests/test_data/players/batting_stats.json", "w") as f:
            json.dump(batting_stats, f, indent=2)
        
        with open("tests/test_data/players/bowling_stats.json", "w") as f:
            json.dump(bowling_stats, f, indent=2)
        
        # Create sample ball-by-ball data
        self._create_bbb_data()
        
        # Create sample venue stats
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
        
        with open("tests/test_data/venues/venue_stats.json", "w") as f:
            json.dump(venue_stats, f, indent=2)
    
    def _create_bbb_data(self):
        """Create sample ball-by-ball data for testing."""
        import csv
        from datetime import datetime, timedelta
        
        # Define constants
        NUM_MATCHES = 3
        VENUES = ['venue1', 'venue2', 'venue3']
        TEAMS = ['team1', 'team2']
        PLAYERS = {
            'team1': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11'],
            'team2': ['p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p22']
        }
        OUTCOMES = ['dot', '1', '2', '3', '4', '6', 'wicket', 'wide', 'no_ball', 'bye', 'leg_bye']
        OUTCOME_WEIGHTS = [40, 30, 8, 2, 10, 5, 3, 1, 0.5, 0.3, 0.2]  # Weights for random selection
        SHOT_TYPES = ['drive', 'cut', 'pull', 'sweep', 'defense', 'loft']
        LINE_LENGTHS = [
            ('good', 'fullish'), ('good', 'length'), ('good', 'short'), 
            ('off', 'fullish'), ('off', 'length'), ('off', 'short'),
            ('leg', 'fullish'), ('leg', 'length'), ('leg', 'short')
        ]
        
        data = []
        base_date = datetime.now() - timedelta(days=30)
        
        # Generate 3 matches between team1 and team2
        for match_id in range(1, NUM_MATCHES + 1):
            match_date = base_date + timedelta(days=match_id)
            venue_id = VENUES[match_id % len(VENUES)]
            
            # Always use team1 and team2
            home_team, away_team = 'team1', 'team2'
            
            # Generate data for both innings
            for innings in [1, 2]:
                if innings == 1:
                    batting_team = home_team
                    bowling_team = away_team
                else:
                    batting_team = away_team
                    bowling_team = home_team
                
                batsmen = PLAYERS[batting_team][:2]  # Start with 2 batsmen
                available_batsmen = PLAYERS[batting_team][2:]
                
                # Innings
                for over in range(1, 21):  # 20 overs
                    # Select a bowler for this over
                    bowler = PLAYERS[bowling_team][over % len(PLAYERS[bowling_team])]
                    
                    for ball in range(1, 7):  # 6 balls per over
                        # Determine outcome of the ball
                        outcome_idx = random.choices(range(len(OUTCOMES)), weights=OUTCOME_WEIGHTS, k=1)[0]
                        outcome = OUTCOMES[outcome_idx]
                        
                        # Additional metadata
                        shot_type = random.choice(SHOT_TYPES) if outcome not in ['wide', 'no_ball', 'bye', 'leg_bye'] else None
                        line, length = random.choice(LINE_LENGTHS)
                        
                        # Determine runs
                        runs = 0
                        is_wicket = False
                        if outcome == 'dot':
                            runs = 0
                        elif outcome in ['1', '2', '3', '4', '6']:
                            runs = int(outcome)
                        elif outcome == 'wicket':
                            runs = 0
                            is_wicket = True
                        elif outcome == 'wide':
                            runs = 1
                        elif outcome == 'no_ball':
                            runs = 1 + random.choice([0, 1, 2, 4, 6])  # No-ball plus possible runs
                        elif outcome in ['bye', 'leg_bye']:
                            runs = random.choice([1, 2, 4])
                        
                        # Determine phase based on over number
                        phase = 1 if over <= 6 else (2 if over <= 12 else (3 if over <= 16 else 4))
                        
                        # Create the ball record
                        ball_record = {
                            'match_id': f'match{match_id}',
                            'match_date': match_date.strftime('%Y-%m-%d'),
                            'venue_id': venue_id,
                            'innings': innings,
                            'over': over,
                            'ball': ball,
                            'phase': phase,  # Add phase to ball record
                            'batting_team': batting_team,
                            'bowling_team': bowling_team,
                            'striker': batsmen[0],
                            'non_striker': batsmen[1],
                            'bowler': bowler,
                            'runs': runs,
                            'is_wicket': 1 if is_wicket else 0,
                            'outcome': outcome,
                            'shot_type': shot_type,
                            'line': line,
                            'length': length
                        }
                        
                        data.append(ball_record)
                        
                        # Handle wicket
                        if is_wicket:
                            # Replace the dismissed batsman if more are available
                            if available_batsmen:
                                dismissed = batsmen[0]
                                new_batsman = available_batsmen.pop(0)
                                batsmen[0] = new_batsman
                            else:
                                # All out - end innings
                                break
                        
                        # Rotate strike for odd runs (except on the last ball of the over)
                        if runs % 2 == 1 and ball < 6:
                            batsmen[0], batsmen[1] = batsmen[1], batsmen[0]
                    
                    # End of over - rotate strike
                    batsmen[0], batsmen[1] = batsmen[1], batsmen[0]
        
        # Write to CSV
        csv_path = "tests/test_data/matches/t20_bbb.csv"
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'match_id', 'match_date', 'venue_id', 'innings', 'over', 'ball',
                'batting_team', 'bowling_team', 'striker', 'non_striker', 'bowler',
                'runs', 'is_wicket', 'outcome', 'shot_type', 'line', 'length', 'phase'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    def test_data_loader(self):
        """Test the DataLoader class."""
        from src.utils.data_loader import DataLoader
        
        loader = DataLoader("tests/test_data")
        
        # Test loading squad data
        squad_data = loader.load_squad_data()
        self.assertIsNotNone(squad_data)
        self.assertIn("team1", squad_data)
        self.assertIn("team2", squad_data)
        
        # Test loading batting stats
        batting_stats = loader.load_player_batting_stats()
        self.assertIsNotNone(batting_stats)
        self.assertIn("p1", batting_stats)
        
        # Test loading bowling stats
        bowling_stats = loader.load_player_bowling_stats()
        self.assertIsNotNone(bowling_stats)
        self.assertIn("p1", bowling_stats)
        
        # Test loading ball-by-ball data
        bbb_data = loader.load_bbb_data()
        self.assertIsNotNone(bbb_data)
        self.assertGreater(len(bbb_data), 0)
        self.assertIn('match_id', bbb_data.columns)
        self.assertIn('batting_team', bbb_data.columns)
        self.assertIn('runs', bbb_data.columns)
        
        # Test processing historical data
        processed_data = loader.process_historical_data()
        self.assertIsNotNone(processed_data)
        self.assertIn('phase', processed_data.columns)
        
        # Test generating venue profiles
        venue_stats = loader.generate_venue_profiles()
        self.assertIsNotNone(venue_stats)
        self.assertGreater(len(venue_stats), 0)
        
        # Test linking data
        linked_data = loader.link_data_sources()
        self.assertIsNotNone(linked_data)
        self.assertIn("teams", linked_data)
        self.assertIn("players", linked_data)
        self.assertIn("venues", linked_data)
    
    def test_player_model(self):
        """Test the Player model."""
        from src.utils.data_loader import DataLoader
        from src.models.player import Player
        
        # Load test data
        loader = DataLoader("tests/test_data")
        linked_data = loader.load_comprehensive_data()
        
        # Create a test player
        player_data = linked_data['players']['p1']
        player = Player("p1", player_data)
        
        # Check basic properties
        self.assertEqual(player.id, "p1")
        self.assertEqual(player.name, "Player 1")
        self.assertEqual(player.main_role, "batsman")
        self.assertIn("opener", player.specific_roles)
        
        # Create a test bowler
        bowler_data = linked_data['players']['p4']
        bowler = Player("p4", bowler_data)
        
        # Test probability generation
        match_state = {
            'innings': 1,
            'score': 100,
            'wickets': 2,
            'overs': 10,
            'current_phase': 2,
            'required_run_rate': 8.0,
            'wickets_remaining': 8,
            'balls_remaining': 60
        }
        
        # Test batting outcome probabilities
        batting_probs = player.get_batting_outcome_probability(bowler, 2, match_state)
        self.assertIsNotNone(batting_probs)
        self.assertTrue(sum(batting_probs.values()) > 0.99)  # Should sum to approximately 1
        
        # Test bowling outcome probabilities
        bowling_probs = bowler.get_bowling_outcome_probability(player, 2, match_state)
        self.assertIsNotNone(bowling_probs)
        self.assertTrue(sum(bowling_probs.values()) > 0.99)  # Should sum to approximately 1
    
    def test_team_model(self):
        """Test the Team model."""
        from src.utils.data_loader import DataLoader
        from src.models.player import Player
        from src.models.team import Team
        
        # Load test data
        loader = DataLoader("tests/test_data")
        linked_data = loader.load_comprehensive_data()
        
        # Create player objects
        player_objects = {}
        for player_id, player_data in linked_data['players'].items():
            player_objects[player_id] = Player(player_id, player_data)
        
        # Create a test team
        team_data = linked_data['teams']['team1']
        team = Team("team1", team_data, player_objects)
        
        # Check basic properties
        self.assertEqual(team.id, "team1")
        self.assertEqual(team.name, "Test Team 1")
        self.assertGreaterEqual(len(team.players), 5)
        
        # Test batting order creation
        batting_order = team.create_batting_order()
        print(f"Batting order: {batting_order}")
        self.assertIsNotNone(batting_order)
        self.assertGreaterEqual(len(batting_order), 3)
        
        # Test bowling rotation creation
        bowling_rotation = team.create_bowling_rotation()
        self.assertIsNotNone(bowling_rotation)
        self.assertEqual(len(bowling_rotation), 20)  # 20 overs in T20
        
        # Test bowler selection
        match_state = {
            'over': 10,
            'bowler_overs': {p: 2 for p in team.players.keys()},
            'current_partnership_runs': 20
        }
        
        selected_bowler = team.select_bowler(10, ["p6", "p7"], match_state)
        self.assertIsNotNone(selected_bowler)
        self.assertIn(selected_bowler, team.players.keys())
    
    def test_match_model(self):
        """Test the Match model."""
        from src.utils.data_loader import DataLoader
        from src.models.player import Player
        from src.models.team import Team
        from src.models.match import Match
        
        # Load test data
        loader = DataLoader("tests/test_data")
        linked_data = loader.load_comprehensive_data()
        
        # Create player objects
        player_objects = {}
        for player_id, player_data in linked_data['players'].items():
            player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        team1 = Team("team1", linked_data['teams']['team1'], player_objects)
        team2 = Team("team2", linked_data['teams']['team2'], player_objects)
        
        # Create a match
        match = Match(team1, team2)
        
        # Test toss simulation
        toss_winner, decision = match.simulate_toss()
        self.assertIsNotNone(toss_winner)
        self.assertIn(decision, ['bat', 'field'])
        
        # Test full match simulation
        match_results = match.simulate_match()
        self.assertIsNotNone(match_results)
        
        # Check basic match results
        self.assertIn('innings', match_results)
        self.assertIn('result', match_results)
        self.assertIn('player_stats', match_results)
        
        # Verify first innings data
        self.assertIn(1, match_results['innings'])
        self.assertIn('score', match_results['innings'][1])
        self.assertIn('wickets', match_results['innings'][1])
        
        # Verify second innings data
        self.assertIn(2, match_results['innings'])
        self.assertIn('score', match_results['innings'][2])
        self.assertIn('wickets', match_results['innings'][2])
        
        # Verify match result
        self.assertIn('winner', match_results['result'])
        self.assertIn('margin', match_results['result'])

        # Display a nicely formatted cricket scorecard
        self._display_match_scorecard(match_results)
    
    def _display_match_scorecard(self, match_results):
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
    
    def test_simulation_engine(self):
        """Test the SimulationEngine."""
        # This test will be limited as we don't want to run multiple simulations in a unit test
        from src.utils.data_loader import DataLoader
        from src.simulation.engine import SimulationEngine
        
        # Load test data
        loader = DataLoader("tests/test_data")
        engine = SimulationEngine(loader, "tests/test_output")
        
        # Test data loading
        player_objects, team_objects, venue_stats = engine.load_comprehensive_data()
        self.assertIsNotNone(player_objects)
        self.assertIsNotNone(team_objects)
        
        # Test batch preparation
        batch_params = engine.prepare_simulation_batch("team1", "team2", None, 2)
        self.assertIsNotNone(batch_params)
        self.assertEqual(batch_params['team1_id'], "team1")
        self.assertEqual(batch_params['team2_id'], "team2")
        self.assertEqual(batch_params['num_simulations'], 2)
    
    def tearDown(self):
        """Clean up after tests."""
        # If you want to keep test data for inspection, comment out these lines
        import shutil
        """if os.path.exists("tests/test_data"):
            shutil.rmtree("tests/test_data")
        if os.path.exists("tests/test_output"):
            shutil.rmtree("tests/test_output")"""


if __name__ == "__main__":
    unittest.main()
