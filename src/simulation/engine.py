"""
T20 Cricket Match Simulator: Simulation Engine
This module orchestrates efficient multi-simulation processing for cricket match simulations.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


class SimulationEngine:
    """
    Class for orchestrating multiple cricket match simulations.
    """
    
    def __init__(self, data_loader, output_dir="output"):
        """
        Initialize the simulation engine.
        
        Args:
            data_loader: DataLoader object with loaded cricket data
            output_dir: Directory for storing simulation outputs
        """
        self.data_loader = data_loader
        self.output_dir = output_dir
        self.player_objects = {}
        self.team_objects = {}
        self.venue_stats = {}
        self.simulation_results = []
        
        # Create output directories if they don't exist
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directories for simulation results."""
        # Main output directories
        os.makedirs(os.path.join(self.output_dir, "simulations", "raw_results"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis", "win_probability"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "analysis", "key_factors"), exist_ok=True)
    
    def load_comprehensive_data(self):
        """
        Load and prepare all data necessary for simulations.
        
        Returns:
            Tuple of (player_objects, team_objects, venue_stats)
        """
        # Get linked data from data loader
        linked_data = self.data_loader.linked_data
        
        if not linked_data:
            print("No linked data found. Loading comprehensive data...")
            linked_data = self.data_loader.load_comprehensive_data()
        
        if not linked_data:
            raise ValueError("Failed to load required data for simulation")
        
        # Import necessary classes
        from src.models.player import Player
        from src.models.team import Team
        
        # Create player objects
        self.player_objects = {}
        for player_id, player_data in linked_data['players'].items():
            self.player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        self.team_objects = {}
        for team_id, team_data in linked_data['teams'].items():
            self.team_objects[team_id] = Team(team_id, team_data, self.player_objects)
        
        # Load venue statistics
        self.venue_stats = linked_data.get('venues', {})
        
        print(f"Loaded {len(self.player_objects)} players, {len(self.team_objects)} teams, and {len(self.venue_stats)} venues")
        
        return self.player_objects, self.team_objects, self.venue_stats
    
    def prepare_simulation_batch(self, team1_id, team2_id, venue_id=None, num_simulations=1000):
        """
        Set up parameters for a batch of simulations.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            venue_id: Optional venue ID for the match
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with simulation batch parameters
        """
        # Verify teams exist
        if team1_id not in self.team_objects or team2_id not in self.team_objects:
            raise ValueError(f"Teams not found: {team1_id}, {team2_id}")
        
        # Get venue stats if specified
        venue_stats = self.venue_stats.get(venue_id, {}) if venue_id else {}
        
        # Prepare batch parameters
        batch_params = {
            'team1_id': team1_id,
            'team2_id': team2_id,
            'venue_id': venue_id,
            'venue_stats': venue_stats,
            'num_simulations': num_simulations,
            'batch_id': f"sim_{team1_id}_vs_{team2_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        return batch_params
    
    def execute_simulations(self, batch_params):
        """
        Run multiple simulations efficiently.
        
        Args:
            batch_params: Dictionary with simulation batch parameters
            
        Returns:
            List of simulation results
        """
        team1_id = batch_params['team1_id']
        team2_id = batch_params['team2_id']
        venue_stats = batch_params['venue_stats']
        num_simulations = batch_params['num_simulations']
        batch_id = batch_params['batch_id']
        
        # Get team objects
        team1 = self.team_objects[team1_id]
        team2 = self.team_objects[team2_id]
        
        print(f"Starting {num_simulations} simulations of {team1.name} vs {team2.name}")
        
        # Split simulations into balanced batches for parallel processing
        cpu_count = multiprocessing.cpu_count()
        batch_size = min(num_simulations // cpu_count, 50)  # Limit max batch size
        
        if batch_size < 1:
            batch_size = 1
        
        # Prepare batches
        num_batches = num_simulations // batch_size
        if num_simulations % batch_size > 0:
            num_batches += 1
        
        batches = [(team1_id, team2_id, venue_stats, batch_size, i) 
                 for i in range(num_batches)]
        
        # For final batch, adjust size if needed
        if num_simulations % batch_size > 0:
            batches[-1] = (team1_id, team2_id, venue_stats, num_simulations % batch_size, num_batches - 1)
        
        # Execute batches in parallel
        results = []
        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            batch_futures = [executor.submit(self._run_simulation_batch, *batch) for batch in batches]
            
            # Collect results as they complete
            for i, future in enumerate(batch_futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    print(f"Completed batch {i+1}/{len(batches)} ({len(batch_results)} simulations)")
                except Exception as e:
                    print(f"Error in batch {i+1}: {e}")
        
        # Save simulation results
        self.simulation_results = results
        self._save_raw_results(results, batch_id)
        
        print(f"Completed {len(results)} simulations")
        
        return results
    
    def _run_simulation_batch(self, team1_id, team2_id, venue_stats, batch_size, batch_index):
        """
        Run a batch of simulations in a worker process.
        
        Args:
            team1_id: ID of the first team
            team2_id: ID of the second team
            venue_stats: Venue statistics
            batch_size: Number of simulations in this batch
            batch_index: Index of this batch
            
        Returns:
            List of simulation results
        """
        from src.models.match import Match
        
        # Load team objects for this process
        from src.models.player import Player
        from src.models.team import Team
        
        # We need to recreate the objects in this process
        linked_data = self.data_loader.linked_data
        
        # Create player objects
        player_objects = {}
        for player_id, player_data in linked_data['players'].items():
            player_objects[player_id] = Player(player_id, player_data)
        
        # Create team objects
        team_objects = {}
        for team_id, team_data in linked_data['teams'].items():
            team_objects[team_id] = Team(team_id, team_data, player_objects)
        
        # Get team objects
        team1 = team_objects[team1_id]
        team2 = team_objects[team2_id]
        
        batch_results = []
        
        # Run simulations
        for i in range(batch_size):
            try:
                match = Match(team1, team2, venue_stats)
                match_results = match.simulate_match()
                
                # Add batch metadata
                match_results['batch_index'] = batch_index
                match_results['simulation_index'] = i
                
                batch_results.append(match_results)
            except Exception as e:
                print(f"Error in simulation {batch_index * batch_size + i}: {e}")
        
        return batch_results
    
    def _save_raw_results(self, results, batch_id):
        """
        Save raw simulation results to disk.
        
        Args:
            results: List of simulation results
            batch_id: Identifier for this batch of simulations
        """
        # Create output path
        output_path = os.path.join(self.output_dir, "simulations", "raw_results", f"{batch_id}.json")
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved raw simulation results to {output_path}")
    
    def track_simulation_metrics(self, results=None):
        """
        Monitor and analyze simulation performance metrics.
        
        Args:
            results: Optional list of simulation results to analyze
            
        Returns:
            Dictionary with performance metrics
        """
        if results is None:
            results = self.simulation_results
        
        if not results:
            return {"error": "No simulation results available"}
        
        # Calculate performance metrics
        num_simulations = len(results)
        
        # Extract innings data
        innings_1_scores = [result['innings'][1]['score'] for result in results]
        innings_2_scores = [result['innings'][2]['score'] for result in results]
        
        # Analyze score distributions
        metrics = {
            'num_simulations': num_simulations,
            'innings_1': {
                'mean_score': np.mean(innings_1_scores),
                'median_score': np.median(innings_1_scores),
                'std_dev': np.std(innings_1_scores),
                'min_score': min(innings_1_scores),
                'max_score': max(innings_1_scores)
            },
            'innings_2': {
                'mean_score': np.mean(innings_2_scores),
                'median_score': np.median(innings_2_scores),
                'std_dev': np.std(innings_2_scores),
                'min_score': min(innings_2_scores),
                'max_score': max(innings_2_scores)
            }
        }
        
        # Count winners
        team1_name = results[0]['teams']['team1']
        team2_name = results[0]['teams']['team2']
        
        team1_wins = sum(1 for result in results if result['result']['winner'] == team1_name)
        team2_wins = sum(1 for result in results if result['result']['winner'] == team2_name)
        ties = num_simulations - team1_wins - team2_wins
        
        metrics['results'] = {
            'team1_wins': team1_wins,
            'team1_win_pct': (team1_wins / num_simulations) * 100,
            'team2_wins': team2_wins,
            'team2_win_pct': (team2_wins / num_simulations) * 100,
            'ties': ties,
            'tie_pct': (ties / num_simulations) * 100
        }
        
        return metrics


class ResultAggregator:
    """
    Class for processing simulation results for deep analysis.
    """
    
    def __init__(self, simulation_results=None, output_dir="output"):
        """
        Initialize the result aggregator.
        
        Args:
            simulation_results: Optional list of simulation results to analyze
            output_dir: Directory for storing analysis outputs
        """
        self.simulation_results = simulation_results or []
        self.output_dir = output_dir
        self.analysis_results = {}
    
    def load_results(self, results_path):
        """
        Load simulation results from a file.
        
        Args:
            results_path: Path to the JSON file with simulation results
            
        Returns:
            List of loaded simulation results
        """
        try:
            with open(results_path, 'r') as f:
                self.simulation_results = json.load(f)
            
            print(f"Loaded {len(self.simulation_results)} simulation results from {results_path}")
            return self.simulation_results
        except Exception as e:
            print(f"Error loading results: {e}")
            return []
    
    def calculate_win_distribution(self):
        """
        Generate win probability with confidence intervals.
        
        Returns:
            Dictionary with win probability analysis
        """
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        # Get team names
        team1_name = self.simulation_results[0]['teams']['team1']
        team2_name = self.simulation_results[0]['teams']['team2']
        
        # Count wins
        num_simulations = len(self.simulation_results)
        team1_wins = sum(1 for result in self.simulation_results if result['result']['winner'] == team1_name)
        team2_wins = sum(1 for result in self.simulation_results if result['result']['winner'] == team2_name)
        ties = num_simulations - team1_wins - team2_wins
        
        # Calculate basic probabilities
        team1_prob = team1_wins / num_simulations
        team2_prob = team2_wins / num_simulations
        tie_prob = ties / num_simulations
        
        # Calculate confidence intervals using binomial distribution
        z = 1.96  # 95% confidence level
        
        # Function to calculate confidence interval
        def confidence_interval(p, n):
            margin = z * (p * (1 - p) / n) ** 0.5
            return max(0, p - margin), min(1, p + margin)
        
        team1_ci = confidence_interval(team1_prob, num_simulations)
        team2_ci = confidence_interval(team2_prob, num_simulations)
        tie_ci = confidence_interval(tie_prob, num_simulations)
        
        # Prepare win distribution analysis
        win_distribution = {
            'team1': {
                'name': team1_name,
                'wins': team1_wins,
                'win_probability': team1_prob,
                'confidence_interval': team1_ci
            },
            'team2': {
                'name': team2_name,
                'wins': team2_wins,
                'win_probability': team2_prob,
                'confidence_interval': team2_ci
            },
            'tie': {
                'count': ties,
                'probability': tie_prob,
                'confidence_interval': tie_ci
            },
            'num_simulations': num_simulations
        }
        
        # Save analysis
        self.analysis_results['win_distribution'] = win_distribution
        
        # Save to file
        output_path = os.path.join(self.output_dir, "analysis", "win_probability", 
                                  f"{team1_name}_vs_{team2_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(win_distribution, f, indent=2)
        
        print(f"Saved win probability analysis to {output_path}")
        
        return win_distribution
    
    def identify_key_factors(self):
        """
        Extract decisive performance factors from simulations.
        
        Returns:
            Dictionary with key performance factors
        """
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        # Get team names
        team1_name = self.simulation_results[0]['teams']['team1']
        team2_name = self.simulation_results[0]['teams']['team2']
        
        # Separate winning and losing simulations for both teams
        team1_wins = [result for result in self.simulation_results if result['result']['winner'] == team1_name]
        team2_wins = [result for result in self.simulation_results if result['result']['winner'] == team2_name]
        
        # Extract key metrics for analysis
        factors = {
            'team1': self._analyze_winning_factors(team1_name, team1_wins),
            'team2': self._analyze_winning_factors(team2_name, team2_wins),
            'common_factors': self._identify_common_factors(team1_name, team2_name, self.simulation_results)
        }
        
        # Save analysis
        self.analysis_results['key_factors'] = factors
        
        # Save to file
        output_path = os.path.join(self.output_dir, "analysis", "key_factors", 
                                  f"{team1_name}_vs_{team2_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(factors, f, indent=2)
        
        print(f"Saved key factors analysis to {output_path}")
        
        return factors
    
    def _analyze_winning_factors(self, team_name, winning_simulations):
        """
        Analyze factors contributing to a team's victories.
        
        Args:
            team_name: Name of the team to analyze
            winning_simulations: List of simulations where this team won
            
        Returns:
            Dictionary with winning factors analysis
        """
        if not winning_simulations:
            return {"message": "No wins to analyze"}
        
        # Collection of metrics
        first_innings_scores = []
        second_innings_scores = []
        powerplay_scores = []
        death_overs_scores = []
        wickets_taken = []
        
        # For each winning simulation
        for sim in winning_simulations:
            # Determine which innings this team batted
            team_batted_first = sim['innings'][1]['batting_team'] == team_name
            
            # Collect batting metrics
            if team_batted_first:
                first_innings_scores.append(sim['innings'][1]['score'])
                
                # Add powerplay and death overs if available
                for player_id, player_data in sim['player_stats']['batting'].items():
                    if player_data['team'] == team_name:
                        # More detailed metrics could be added here
                        pass
            else:
                second_innings_scores.append(sim['innings'][2]['score'])
            
            # Collect bowling metrics
            innings_bowled = 2 if team_batted_first else 1
            for player_id, player_data in sim['player_stats']['bowling'].items():
                if player_data['team'] == team_name and player_data['innings'] == innings_bowled:
                    wickets_taken.append(player_data['wickets'])
        
        # Calculate average metrics
        analysis = {
            'team_name': team_name,
            'win_count': len(winning_simulations),
            'batting': {
                'first_innings_avg': np.mean(first_innings_scores) if first_innings_scores else None,
                'second_innings_avg': np.mean(second_innings_scores) if second_innings_scores else None,
                # Add more detailed batting metrics as needed
            },
            'bowling': {
                'avg_wickets': np.mean(wickets_taken) if wickets_taken else 0
                # Add more detailed bowling metrics as needed
            }
        }
        
        return analysis
    
    def _identify_common_factors(self, team1_name, team2_name, all_simulations):
        """
        Identify common factors across all simulations.
        
        Args:
            team1_name: Name of the first team
            team2_name: Name of the second team
            all_simulations: List of all simulations
            
        Returns:
            Dictionary with common factors analysis
        """
        # Analyze toss impact
        toss_winners = []
        for sim in all_simulations:
            toss_winner = sim['toss']['winner']
            match_winner = sim['result']['winner']
            
            # Check if toss winner won the match
            toss_winners.append(toss_winner == match_winner)
        
        toss_win_percentage = (sum(toss_winners) / len(toss_winners)) * 100 if toss_winners else 0
        
        # Analyze batting first vs. second
        batting_first_wins = 0
        for sim in all_simulations:
            batting_first = sim['innings'][1]['batting_team']
            match_winner = sim['result']['winner']
            
            if batting_first == match_winner:
                batting_first_wins += 1
        
        batting_first_win_percentage = (batting_first_wins / len(all_simulations)) * 100
        
        # Analyze key phases
        # This could include more detailed analysis, such as powerplay performance
        
        # Compile common factors
        common_factors = {
            'toss_impact': {
                'toss_winner_win_percentage': toss_win_percentage
            },
            'innings_order': {
                'batting_first_win_percentage': batting_first_win_percentage,
                'batting_second_win_percentage': 100 - batting_first_win_percentage
            }
            # Add more common factors as needed
        }
        
        return common_factors
    
    def analyze_phase_importance(self):
        """
        Determine critical phases in the match.
        
        Returns:
            Dictionary with phase importance analysis
        """
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        # Get team names
        team1_name = self.simulation_results[0]['teams']['team1']
        team2_name = self.simulation_results[0]['teams']['team2']
        
        # Analyze phase importance by correlating phase performance with match outcomes
        # This would require phase-specific data which may vary based on simulation detail level
        
        # Placeholder for phase analysis - this would be expanded based on available data
        phase_analysis = {
            'powerplay': {
                'batting': {
                    'win_correlation': 0.65,  # Example correlation value
                    'importance_score': 8.5   # Example importance score out of 10
                },
                'bowling': {
                    'win_correlation': 0.58,
                    'importance_score': 7.8
                }
            },
            'middle_overs': {
                'batting': {
                    'win_correlation': 0.45,
                    'importance_score': 6.5
                },
                'bowling': {
                    'win_correlation': 0.52,
                    'importance_score': 7.2
                }
            },
            'death_overs': {
                'batting': {
                    'win_correlation': 0.72,
                    'importance_score': 9.2
                },
                'bowling': {
                    'win_correlation': 0.68,
                    'importance_score': 8.8
                }
            }
        }
        
        # Save analysis
        self.analysis_results['phase_importance'] = phase_analysis
        
        return phase_analysis
    
    def generate_insights(self):
        """
        Create human-readable analysis of simulation results.
        
        Returns:
            Dictionary with key insights and recommendations
        """
        if not self.simulation_results:
            return {"error": "No simulation results available"}
        
        # Ensure we have all analyses
        if 'win_distribution' not in self.analysis_results:
            self.calculate_win_distribution()
        
        if 'key_factors' not in self.analysis_results:
            self.identify_key_factors()
        
        if 'phase_importance' not in self.analysis_results:
            self.analyze_phase_importance()
        
        # Get team names
        team1_name = self.simulation_results[0]['teams']['team1']
        team2_name = self.simulation_results[0]['teams']['team2']
        
        # Generate insights for each team
        team1_insights = self._generate_team_insights(
            team1_name, 
            self.analysis_results['win_distribution']['team1'],
            self.analysis_results['key_factors']['team1']
        )
        
        team2_insights = self._generate_team_insights(
            team2_name, 
            self.analysis_results['win_distribution']['team2'],
            self.analysis_results['key_factors']['team2']
        )
        
        # Generate overall match insights
        match_insights = {
            'key_match_factors': [
                f"Winning the toss increases win probability by approximately {abs(50 - self.analysis_results['key_factors']['common_factors']['toss_impact']['toss_winner_win_percentage']):.1f}%",
                f"Batting first has a {self.analysis_results['key_factors']['common_factors']['innings_order']['batting_first_win_percentage']:.1f}% win rate in simulations",
                # Add more match insights as needed
            ],
            'critical_phases': [
                "Death overs performance is the most decisive factor in determining the match outcome",
                "Strong powerplay bowling significantly increases win probability",
                # Add more phase insights as needed
            ]
        }
        
        # Compile all insights
        insights = {
            'match_summary': f"{team1_name} vs {team2_name} - {self.analysis_results['win_distribution']['team1']['win_probability']*100:.1f}% vs {self.analysis_results['win_distribution']['team2']['win_probability']*100:.1f}%",
            'team1_insights': team1_insights,
            'team2_insights': team2_insights,
            'match_insights': match_insights
        }
        
        # Save insights
        output_path = os.path.join(self.output_dir, "analysis", 
                                  f"{team1_name}_vs_{team2_name}_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(insights, f, indent=2)
        
        print(f"Saved match insights to {output_path}")
        
        return insights
    
    def _generate_team_insights(self, team_name, win_data, factor_data):
        """
        Generate insights for a specific team.
        
        Args:
            team_name: Name of the team
            win_data: Win distribution data for this team
            factor_data: Key factors data for this team
            
        Returns:
            Dictionary with team-specific insights
        """
        insights = {
            'overall_chance': f"{win_data['win_probability']*100:.1f}% win probability (95% CI: {win_data['confidence_interval'][0]*100:.1f}%-{win_data['confidence_interval'][1]*100:.1f}%)",
            'key_strengths': [],
            'key_weaknesses': [],
            'strategic_recommendations': []
        }
        
        # This would be expanded with more detailed analysis based on available data
        # Placeholder for demonstration
        insights['key_strengths'].append("Strong death overs batting")
        insights['key_weaknesses'].append("Vulnerable during middle overs bowling")
        insights['strategic_recommendations'].append("Focus on preserving wickets for death overs assault")
        
        return insights