"""
Test script to display team rankings and orders from the T20 Cricket Match Simulator.
"""

import os
import sys
import json
import unittest
from typing import Dict, List, Tuple, Any

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import DataLoader
from src.models.player import Player
from src.models.team import Team

class TeamRankingsTest(unittest.TestCase):
    """Test class for displaying team rankings and orders."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test data directory if it doesn't exist
        os.makedirs("tests/advanced_test_data", exist_ok=True)
        
        # Load test data
        self.loader = DataLoader("tests/advanced_test_data")
        self.linked_data = self.loader.load_comprehensive_data()
        
        # Create player objects
        self.player_objects = {}
        for player_id, player_data in self.linked_data['players'].items():
            self.player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        self.team_objects = {}
        for team_id, team_data in self.linked_data['teams'].items():
            team = Team(team_id, team_data, self.player_objects)
            
            # Validate team has required components
            self._validate_team(team)
            
            # Get optimal XI and orders
            try:
                team.optimal_xi, team.optimal_batting_order, team.optimal_bowling_allocation = team.pick_optimal_xi()
            except ZeroDivisionError:
                print(f"\nWarning: Could not optimize team {team.name} due to insufficient bowling data")
                # Set default values
                team.optimal_xi = list(team.players.keys())[:11]  # First 11 players
                team.optimal_batting_order = team.optimal_xi.copy()
                team.optimal_bowling_allocation = {i: team.optimal_xi[i % len(team.optimal_xi)] 
                                                 for i in range(1, 21)}
            
            self.team_objects[team_id] = team

    def _validate_team(self, team: Team):
        """Validate that a team has the required components."""
        # Check if team has players
        self.assertGreater(len(team.players), 0, f"Team {team.name} has no players")
        
        # Check if team has batting rankings
        self.assertIsNotNone(team.batting_position_rankings, 
                            f"Team {team.name} has no batting position rankings")
        
        # Check if team has bowling rankings
        self.assertIsNotNone(team.bowling_over_rankings, 
                            f"Team {team.name} has no bowling over rankings")
        
        # Check if team has at least 11 players
        self.assertGreaterEqual(len(team.players), 11, 
                              f"Team {team.name} has fewer than 11 players")

    def display_team_rankings(self, team: Team):
        """Display batting position and bowling over rankings for a team."""
        print("\n" + "="*80)
        print(f"{team.name.upper()} RANKINGS")
        print("="*80)
        
        # Display batting position rankings
        print("\nBATTING POSITION RANKINGS:")
        print("-"*80)
        print(f"{'Position':<10} {'Player':<25} {'Score':<10} {'Roles':<20}")
        print("-"*80)
        
        for position in range(1, 12):
            rankings = team.batting_position_rankings.at[position, 'rankings']
            if rankings:
                top_3 = rankings[:3]  # Show top 3 players for each position
                for player in top_3:
                    print(f"{position:<10} {player['name']:<25} {player['score']:<10.2f} {', '.join(player['batting_roles']):<20}")
        
        # Display bowling over rankings
        print("\nBOWLING OVER RANKINGS:")
        print("-"*80)
        print(f"{'Over':<10} {'Bowler':<25} {'Score':<10} {'Roles':<20}")
        print("-"*80)
        
        for over in range(1, 21):
            rankings = team.bowling_over_rankings.at[over, 'rankings']
            if rankings:
                top_3 = rankings[:3]  # Show top 3 bowlers for each over
                for bowler in top_3:
                    print(f"{over:<10} {bowler['name']:<25} {bowler['score']:<10.2f} {', '.join(bowler['bowling_roles']):<20}")

    def display_team_orders(self, team: Team):
        """Display optimal batting and bowling orders for a team."""
        print("\n" + "="*80)
        print(f"{team.name.upper()} OPTIMAL ORDERS")
        print("="*80)
        
        # Display batting order
        print("\nOPTIMAL BATTING ORDER:")
        print("-"*80)
        print(f"{'Position':<10} {'Player':<25} {'Roles':<20}")
        print("-"*80)
        
        for pos, player_id in enumerate(team.optimal_batting_order, 1):
            if player_id:
                player = self.player_objects[player_id]
                roles = team.squad[player_id]['batting_roles']
                print(f"{pos:<10} {player.name:<25} {', '.join(roles):<20}")
        
        # Display bowling order
        print("\nOPTIMAL BOWLING ORDER:")
        print("-"*80)
        print(f"{'Over':<10} {'Bowler':<25} {'Roles':<20}")
        print("-"*80)
        
        for over, bowler_id in team.optimal_bowling_allocation.items():
            if bowler_id:
                bowler = self.player_objects[bowler_id]
                roles = team.squad[bowler_id]['bowling_roles']
                print(f"{over:<10} {bowler.name:<25} {', '.join(roles):<20}")

    def test_team_rankings_and_orders(self):
        """Test displaying rankings and orders for all teams."""
        for team_id, team in self.team_objects.items():
            # Display rankings
            self.display_team_rankings(team)
            
            # Display orders
            self.display_team_orders(team)
            
            print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    unittest.main()