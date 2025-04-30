"""
Setup script for T20 Cricket Match Simulator.
This script helps install dependencies and set up the project structure.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path
import json


def setup_environment():
    """Install required dependencies."""
    print("Installing required dependencies...")
    
    # Required packages
    requirements = [
        "numpy>=1.22.0",
        "pandas>=1.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0"
    ]
    
    # Install packages
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("Dependencies installed successfully!")


def create_directory_structure():
    """Create the basic project directory structure."""
    print("Setting up project directory structure...")
    
    # Main directories
    directories = [
        "data/teams",
        "data/players",
        "data/matches",
        "data/venues",
        "data/processed",
        "output/simulations/raw_results",
        "output/analysis/win_probability",
        "output/analysis/key_factors",
        "src/models",
        "src/simulation",
        "src/utils",
        "src/visualization",
        "tests/unit",
        "tests/integration",
        "tests/performance"
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create __init__.py files
    python_dirs = [
        "src",
        "src/models",
        "src/simulation",
        "src/utils",
        "src/visualization",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/performance"
    ]
    
    for directory in python_dirs:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            Path(init_file).touch()
    
    print("Directory structure created successfully!")


def create_sample_files():
    """Create sample data files."""
    print("Creating sample data files...")
    
    # Sample squad profiles
    squad_data = {
        "team1": {
            "name": "Sample Team 1",
            "players": [
                {"player_id": "p1", "name": "Player 1", "main_role": "batsman", "specific_roles": ["opener"]},
                {"player_id": "p2", "name": "Player 2", "main_role": "batsman", "specific_roles": ["middle-order"]},
                {"player_id": "p3", "name": "Player 3", "main_role": "all-rounder", "specific_roles": ["finisher"]},
                {"player_id": "p4", "name": "Player 4", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]},
                {"player_id": "p5", "name": "Player 5", "main_role": "bowler", "specific_roles": ["death-specialist"]}
            ]
        },
        "team2": {
            "name": "Sample Team 2",
            "players": [
                {"player_id": "p6", "name": "Player 6", "main_role": "batsman", "specific_roles": ["opener"]},
                {"player_id": "p7", "name": "Player 7", "main_role": "batsman", "specific_roles": ["middle-order"]},
                {"player_id": "p8", "name": "Player 8", "main_role": "all-rounder", "specific_roles": ["finisher"]},
                {"player_id": "p9", "name": "Player 9", "main_role": "bowler", "specific_roles": ["powerplay-specialist"]},
                {"player_id": "p10", "name": "Player 10", "main_role": "bowler", "specific_roles": ["death-specialist"]}
            ]
        }
    }
    
    # Write squad data
    with open("data/teams/squad_profiles.json", "w") as f:
        json.dump(squad_data, f, indent=4)
    
    # Sample batting stats template
    batting_stats = {}
    for i in range(1, 11):
        player_id = f"p{i}"
        batting_stats[player_id] = {
            "bat_hand": "right",
            "batter_id": player_id,
            "total_runs": 500,
            "total_balls": 400,
            "dismissals": 20,
            "by_phase": {
                "1": {"runs": 200, "balls": 150, "fours": 10, "sixes": 5, "dots": 80, "dismissals": 5},
                "2": {"runs": 200, "balls": 150, "fours": 10, "sixes": 5, "dots": 80, "dismissals": 5},
                "3": {"runs": 100, "balls": 100, "fours": 5, "sixes": 5, "dots": 40, "dismissals": 10}
            },
            "vs_bowler_styles": {
                "fast": {"runs": 250, "balls": 200, "dismissals": 10},
                "spin": {"runs": 250, "balls": 200, "dismissals": 10}
            }
        }
    
    # Write batting stats
    with open("data/players/batting_stats.json", "w") as f:
        json.dump(batting_stats, f, indent=4)
    
    # Sample bowling stats template
    bowling_stats = {}
    for i in range(1, 11):
        player_id = f"p{i}"
        bowling_stats[player_id] = {
            "bowling_style": "fast" if i % 2 == 0 else "spin",
            "bowler_id": player_id,
            "runs_conceded": 400,
            "balls_bowled": 300,
            "wickets": 20,
            "by_phase": {
                "1": {"runs_conceded": 150, "balls_bowled": 100, "wickets": 5, "dots": 50},
                "2": {"runs_conceded": 150, "balls_bowled": 100, "wickets": 5, "dots": 50},
                "3": {"runs_conceded": 100, "balls_bowled": 100, "wickets": 10, "dots": 40}
            },
            "vs_batsman_types": {
                "aggressive": {"runs_conceded": 200, "balls_bowled": 150, "wickets": 10},
                "anchor": {"runs_conceded": 200, "balls_bowled": 150, "wickets": 10}
            }
        }
    
    # Write bowling stats
    with open("data/players/bowling_stats.json", "w") as f:
        json.dump(bowling_stats, f, indent=4)
    
    # Sample venue stats
    venue_stats = {
        "venue1": {
            "first_innings_avg_score": 165,
            "first_innings_std_score": 25,
            "second_innings_avg_score": 155,
            "second_innings_std_score": 30,
            "matches_played": 50,
            "phase_1_run_rate": 8.2,
            "phase_1_wicket_rate": 0.7,
            "phase_2_run_rate": 7.5,
            "phase_2_wicket_rate": 1.1,
            "phase_3_run_rate": 10.5,
            "phase_3_wicket_rate": 1.8
        }
    }
    
    # Write venue stats
    with open("data/venues/venue_stats.json", "w") as f:
        json.dump(venue_stats, f, indent=4)
    
    print("Sample data files created successfully!")


def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(description="Setup T20 Cricket Match Simulator")
    parser.add_argument("--skip-deps", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--skip-dirs", action="store_true", help="Skip creating directory structure")
    parser.add_argument("--skip-samples", action="store_true", help="Skip creating sample files")
    
    args = parser.parse_args()
    
    print("T20 Cricket Match Simulator Setup")
    print("=================================")
    
    # Install dependencies
    if not args.skip_deps:
        setup_environment()
    
    # Create directory structure
    if not args.skip_dirs:
        create_directory_structure()
    
    # Create sample files
    if not args.skip_samples:
        create_sample_files()
    
    print("\nSetup completed successfully!")
    print("\nTo run a single match simulation:")
    print("python main.py --team1 team1 --team2 team2 --mode single")
    
    print("\nTo run a batch of simulations:")
    print("python main.py --team1 team1 --team2 team2 --simulations 100")


if __name__ == "__main__":
    main()
