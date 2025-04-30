"""
T20 Cricket Match Simulator: Data Loader Module
This module handles the loading and processing of various data sources required for the cricket match simulation.
"""

import json
import pandas as pd
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional


class DataLoader:
    """
    Class for loading, processing, and linking various cricket data sources.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Base directory containing all data files
        """
        self.data_dir = data_dir
        self.squad_data = None
        self.batting_stats = None
        self.bowling_stats = None
        self.bbb_data = None
        self.venue_stats = None
        self.linked_data = None
    
    def load_squad_data(self, file_path: Optional[str] = None) -> Dict:
        """
        Load team squad data from JSON file.
        
        Args:
            file_path: Path to the squad profiles JSON file
            
        Returns:
            Dictionary containing squad data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "teams", "squad_profiles.json")
        
        try:
            with open(file_path, 'r') as f:
                self.squad_data = json.load(f)
            print(f"Successfully loaded squad data from {file_path}")
            return self.squad_data
        except Exception as e:
            print(f"Error loading squad data: {e}")
            return {}
    
    def load_player_batting_stats(self, file_path: Optional[str] = None) -> Dict:
        """
        Load comprehensive batting statistics for all players.
        
        Args:
            file_path: Path to the batting stats JSON file
            
        Returns:
            Dictionary containing batting statistics
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "players", "batting_stats.json")
        
        try:
            with open(file_path, 'r') as f:
                self.batting_stats = json.load(f)
            print(f"Successfully loaded batting stats from {file_path}")
            return self.batting_stats
        except Exception as e:
            print(f"Error loading batting stats: {e}")
            return {}
    
    def load_player_bowling_stats(self, file_path: Optional[str] = None) -> Dict:
        """
        Load comprehensive bowling statistics for all players.
        
        Args:
            file_path: Path to the bowling stats JSON file
            
        Returns:
            Dictionary containing bowling statistics
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "players", "bowling_stats.json")
        
        try:
            with open(file_path, 'r') as f:
                self.bowling_stats = json.load(f)
            print(f"Successfully loaded bowling stats from {file_path}")
            return self.bowling_stats
        except Exception as e:
            print(f"Error loading bowling stats: {e}")
            return {}
    
    def load_bbb_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load ball-by-ball data from CSV file.
        
        Args:
            file_path: Path to the ball-by-ball CSV file
            
        Returns:
            DataFrame containing ball-by-ball data
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "matches", "t20_bbb.csv")
        
        try:
            self.bbb_data = pd.read_csv(file_path)
            print(f"Successfully loaded {len(self.bbb_data)} ball-by-ball records from {file_path}")
            return self.bbb_data
        except Exception as e:
            print(f"Error loading ball-by-ball data: {e}")
            return pd.DataFrame()
    
    def process_historical_data(self) -> pd.DataFrame:
        """
        Process historical ball-by-ball data to extract patterns and statistics.
        
        Returns:
            Processed DataFrame with additional analytical columns
        """
        if self.bbb_data is None:
            print("No ball-by-ball data loaded. Please load data first.")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying the original
        processed_data = self.bbb_data.copy()
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['runs', 'is_wicket', 'over', 'ball', 'innings']
        for col in numeric_columns:
            if col in processed_data.columns:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
        
        # Add 'phase' column based on over number
        def classify_phase(over):
            try:
                over_num = float(over)
                if 0 < over_num <= 6:
                    return 1  # Powerplay
                elif 6 < over_num <= 15:
                    return 2  # Middle overs
                elif 15 < over_num <= 20:
                    return 3  # Death overs
                else:
                    return 0  # Invalid over
            except (ValueError, TypeError):
                return 0  # Invalid over
        
        processed_data['phase'] = processed_data['over'].apply(classify_phase)
        
        # Add additional analytical columns
        if 'runs' in processed_data.columns and 'is_wicket' in processed_data.columns:
            try:
                # Group by match and innings for calculations
                grouped = processed_data.groupby(['match_id', 'innings'])
                
                # Calculate running total score and wickets
                processed_data['running_score'] = grouped['runs'].cumsum()
                processed_data['running_wickets'] = grouped['is_wicket'].cumsum()
                
                # Calculate balls remaining
                processed_data['total_balls'] = grouped['over'].transform('count')
                
                # Create an index column if needed for balls remaining calculation
                if 'ball_number' not in processed_data.columns:
                    processed_data['ball_number'] = processed_data.groupby(['match_id', 'innings']).cumcount() + 1
                
                processed_data['balls_remaining'] = processed_data['total_balls'] - processed_data['ball_number']
                
                print("Successfully processed historical ball-by-ball data")
            except Exception as e:
                print(f"Error processing ball-by-ball data: {e}")
        
        return processed_data
    
    def generate_venue_profiles(self) -> Dict:
        """
        Create statistical profiles for each venue based on historical data.
        
        Returns:
            Dictionary containing venue statistics
        """
        if self.bbb_data is None:
            print("No ball-by-ball data loaded. Please load data first.")
            return {}
        
        venue_stats = {}
        
        # Check if venue_id column exists
        if 'venue_id' not in self.bbb_data.columns:
            print("No venue_id column found in ball-by-ball data")
            return venue_stats
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['runs', 'is_wicket']
        for col in numeric_columns:
            if col in self.bbb_data.columns:
                self.bbb_data[col] = pd.to_numeric(self.bbb_data[col], errors='coerce')
        
        # Group by venue
        venue_groups = self.bbb_data.groupby('venue_id')
        
        for venue_id, group in venue_groups:
            # Get basic venue stats
            innings_data = group.groupby(['match_id', 'innings'])
            
            # Calculate innings totals
            innings_totals = innings_data.agg({
                'runs': 'sum',
                'is_wicket': 'sum'
            })
            
            # Calculate first and second innings stats
            # Need to handle potential missing innings
            first_innings = None
            second_innings = None
            
            try:
                if 1 in innings_totals.index.get_level_values('innings'):
                    first_innings = innings_totals.xs(1, level='innings')
                if 2 in innings_totals.index.get_level_values('innings'):
                    second_innings = innings_totals.xs(2, level='innings')
            except Exception as e:
                print(f"Error analyzing innings data for venue {venue_id}: {e}")
            
            # Store venue statistics
            venue_stats[venue_id] = {
                'first_innings_avg_score': first_innings['runs'].mean() if first_innings is not None and not first_innings.empty else 0,
                'first_innings_std_score': first_innings['runs'].std() if first_innings is not None and not first_innings.empty else 0,
                'second_innings_avg_score': second_innings['runs'].mean() if second_innings is not None and not second_innings.empty else 0,
                'second_innings_std_score': second_innings['runs'].std() if second_innings is not None and not second_innings.empty else 0,
                'matches_played': len(innings_data.groups)
            }
            
            # Add phase-specific statistics if phase column exists
            if 'phase' in group.columns:
                try:
                    phase_stats = group.groupby('phase').agg({
                        'runs': 'sum',
                        'is_wicket': 'sum',
                        'over': 'count'  # count balls/overs
                    })
                    
                    for phase_idx, stats in phase_stats.iterrows():
                        try:
                            phase = int(phase_idx)  # Ensure phase is an integer
                            balls = stats['over']
                            venue_stats[venue_id][f'phase_{phase}_run_rate'] = (stats['runs'] / balls) * 6 if balls > 0 else 0
                            venue_stats[venue_id][f'phase_{phase}_wicket_rate'] = (stats['is_wicket'] / balls) * 6 if balls > 0 else 0
                        except (ValueError, TypeError) as e:
                            print(f"Error processing phase {phase_idx} for venue {venue_id}: {e}")
                except Exception as e:
                    print(f"Error calculating phase statistics for venue {venue_id}: {e}")
        
        self.venue_stats = venue_stats
        print(f"Generated statistical profiles for {len(venue_stats)} venues")
        return venue_stats
    
    def link_data_sources(self) -> Dict:
        """
        Connect all data sources by player/team IDs.
        
        Returns:
            Dictionary with linked data structure
        """
        if not all([self.squad_data, self.batting_stats, self.bowling_stats]):
            print("Missing required data. Please load all data sources first.")
            return {}
        
        linked_data = {
            'teams': {},
            'players': {},
            'venues': self.venue_stats if self.venue_stats else {}
        }
        
        # Process teams and players
        for team_id, team_info in self.squad_data.items():
            team_name = team_info.get('name', f'Team {team_id}')
            players = team_info.get('players', [])
            
            # Initialize team in linked data
            linked_data['teams'][team_id] = {
                'name': team_name,
                'players': []
            }
            
            # Process each player
            for player in players:
                player_id = player.get('player_id')
                
                if not player_id:
                    continue
                
                # Create player entry with basic info
                player_data = {
                    'id': player_id,
                    'name': player.get('name', f'Player {player_id}'),
                    'main_role': player.get('main_role', 'unknown'),
                    'specific_roles': player.get('specific_roles', []),
                    'team_id': team_id
                }
                
                # Link batting stats if available
                if player_id in self.batting_stats:
                    player_data['batting_stats'] = self.batting_stats[player_id]
                
                # Link bowling stats if available
                if player_id in self.bowling_stats:
                    player_data['bowling_stats'] = self.bowling_stats[player_id]
                
                # Add player to the players dictionary
                linked_data['players'][player_id] = player_data
                
                # Add player ID to team's player list
                linked_data['teams'][team_id]['players'].append(player_id)
        
        self.linked_data = linked_data
        print(f"Successfully linked data for {len(linked_data['teams'])} teams and {len(linked_data['players'])} players")
        return linked_data
    
    def load_comprehensive_data(self) -> Dict:
        """
        Load and link all data sources in one operation.
        
        Returns:
            Comprehensive linked data dictionary
        """
        self.load_squad_data()
        self.load_player_batting_stats()
        self.load_player_bowling_stats()
        self.load_bbb_data()
        self.process_historical_data()
        self.generate_venue_profiles()
        return self.link_data_sources()
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save processed and linked data to disk for faster loading.
        
        Args:
            output_path: Path to save processed data
        """
        if self.linked_data is None:
            print("No linked data available. Please process data first.")
            return
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(self.linked_data, f)
            print(f"Successfully saved processed data to {output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")
    
    def load_processed_data(self, input_path: str) -> Dict:
        """
        Load previously processed and linked data from disk.
        
        Args:
            input_path: Path to load processed data from
            
        Returns:
            Loaded data dictionary
        """
        try:
            with open(input_path, 'rb') as f:
                self.linked_data = pickle.load(f)
            print(f"Successfully loaded processed data from {input_path}")
            return self.linked_data
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    comprehensive_data = loader.load_comprehensive_data()
    
    # Save processed data for future use
    loader.save_processed_data("data/processed/linked_data.pkl")
