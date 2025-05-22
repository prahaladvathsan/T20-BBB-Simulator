import pandas as pd
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple, Set
from fuzzywuzzy import fuzz
from jellyfish import jaro_winkler_similarity, metaphone

def normalize_name(name: str) -> str:
    """Normalize player names for better matching."""
    # Convert to lowercase
    name = name.lower()
    # Remove special characters and extra spaces
    name = re.sub(r'[^a-z\s]', '', name)
    # Remove extra spaces
    name = ' '.join(name.split())
    # Remove middle initials (e.g., "John A Smith" -> "John Smith")
    name = re.sub(r'\s+[a-z]\s+', ' ', name)
    return name

def calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate similarity between two names using multiple metrics."""
    # Normalize both names
    norm_name1 = normalize_name(name1)
    norm_name2 = normalize_name(name2)
    
    # Calculate different similarity metrics
    levenshtein_ratio = fuzz.ratio(norm_name1, norm_name2) / 100.0
    token_sort_ratio = fuzz.token_sort_ratio(norm_name1, norm_name2) / 100.0
    jaro_winkler = jaro_winkler_similarity(norm_name1, norm_name2)
    
    # Calculate phonetic similarity
    metaphone1 = metaphone(norm_name1)
    metaphone2 = metaphone(norm_name2)
    phonetic_similarity = 1.0 if metaphone1 == metaphone2 else 0.0
    
    # Weighted combination of metrics
    weights = {
        'levenshtein': 0.3,
        'token_sort': 0.3,
        'jaro_winkler': 0.3,
        'phonetic': 0.1
    }
    
    final_score = (
        weights['levenshtein'] * levenshtein_ratio +
        weights['token_sort'] * token_sort_ratio +
        weights['jaro_winkler'] * jaro_winkler +
        weights['phonetic'] * phonetic_similarity
    )
    
    return final_score * 100  # Convert to percentage for consistency

def get_all_players_from_bbb() -> Dict[str, str]:
    """Extract unique player codes and names from BBB data."""
    # Check if cached file exists
    cache_file = Path('data/teams/all_players.csv')
    if cache_file.exists():
        print("Loading cached player list...")
        df = pd.read_csv(cache_file)
        return dict(zip(df['code'], df['name']))
    
    print("Reading BBB data...")
    bbb_df = pd.read_csv('data/matches/t20_bbb.csv')
    
    # Get unique batsmen
    batsmen = bbb_df[['p_bat', 'bat']].drop_duplicates()
    batsmen_dict = dict(zip(batsmen['p_bat'], batsmen['bat']))
    
    # Get unique bowlers
    bowlers = bbb_df[['p_bowl', 'bowl']].drop_duplicates()
    bowlers_dict = dict(zip(bowlers['p_bowl'], bowlers['bowl']))
    
    # Combine both dictionaries removing duplicates
    all_players = {**batsmen_dict, **bowlers_dict}
    all_players = {k: v for k, v in all_players.items() if k not in all_players.values()}
    
    # Cache the results
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'code': list(all_players.keys()), 'name': list(all_players.values())}).to_csv(cache_file, index=False)
    
    print(f"Found {len(all_players)} unique players in BBB data")
    return all_players

def find_player_code(player_name: str, all_players: Dict[str, str], threshold: float = 0.7) -> List[Tuple[str, str, float]]:
    """Find matching player codes for a given name using multiple similarity metrics."""
    matches = []
    
    # Calculate similarity with all players
    for code, name in all_players.items():
        similarity = calculate_name_similarity(player_name, name)
        if similarity >= threshold * 100:  # Convert threshold to percentage
            matches.append((code, name, similarity))
    
    # Sort by similarity score
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Always return at least the best match
    if not matches:
        best_match = max(all_players.items(), 
                        key=lambda x: calculate_name_similarity(player_name, x[1]))
        matches = [(best_match[0], best_match[1], 
                   calculate_name_similarity(player_name, best_match[1]))]
    
    return matches

def create_squad_profiles(squads_df: pd.DataFrame, all_players: Dict[str, str]) -> Dict:
    """Create squad profiles from squads list and BBB data."""
    squad_profiles = {}
    all_matches = []  # Store all matches for reporting
    
    # Group players by team
    for team_code in squads_df['Team_Code'].unique():
        team_data = squads_df[squads_df['Team_Code'] == team_code]
        team_name = team_data['Team'].iloc[0]
        
        squad_profiles[team_code] = {
            "team_name": team_name,
            "edition": "2024",  # You might want to make this configurable
            "players": []  # Changed from dictionary to list
        }
        
        # Process each player in the team
        for _, player in team_data.iterrows():
            matches = find_player_code(player['Name'], all_players)
            
            # Always use the best match
            code, name, score = matches[0]
            
            # Create player object with player_id field
            player_obj = {
                "player_id": int(code),  # Convert to integer
                "name": name,
                "main_role": player['Role']
            }
            
            # Add player to the list
            squad_profiles[team_code]["players"].append(player_obj)
            
            # Filter other matches to only those within 20 points of best match and limit to 3
            other_matches = []
            for match in matches[1:]:
                if match[2] >= score - 20 and score > 90:
                    other_matches.append(match)
                elif match[2] >= 40:
                    other_matches.append(match)
                    
            
            # Store all matches for reporting
            all_matches.append({
                "team": team_name,
                "squad_name": player['Name'],
                "chosen_match": matches[0],
                "other_matches": other_matches
            })
    
    return squad_profiles, all_matches

def main():
    # Create output directory if it doesn't exist
    output_dir = Path('data/teams')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all players from BBB data
    all_players = get_all_players_from_bbb()
    
    # Read squads list
    print("Reading squads list...")
    squads_df = pd.read_csv('data/teams/IPL_Squads_List.csv')
    
    # Create squad profiles
    print("Creating squad profiles...")
    squad_profiles, all_matches = create_squad_profiles(squads_df, all_players)
    
    # Save squad profiles
    output_file = output_dir / 'squad_profiles.json'
    with open(output_file, 'w') as f:
        json.dump(squad_profiles, f, indent=2)
    print(f"Saved squad profiles to {output_file}")
    
    # Report all matches
    print("\nAll player matches:")
    for match in all_matches:
        print(f"Squad name: {match['squad_name']}")
        print(f"Chosen match: {match['chosen_match'][1]} (code: {match['chosen_match'][0]}, score: {match['chosen_match'][2]:.1f})")
        if match['other_matches']:
            print("Other matches:")
            for code, name, score in match['other_matches']:
                print(f"  - {name} (code: {code}, score: {score:.1f})")

if __name__ == "__main__":
    main()