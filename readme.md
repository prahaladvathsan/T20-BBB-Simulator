# T20 Cricket Match Simulator

A comprehensive simulator for T20 cricket matches that calculates win probabilities across multiple simulations using detailed player and team statistics.

## Overview

This project implements a data-driven T20 cricket match simulator that:
- Leverages rich player and team statistics
- Simulates matches with phase-specific and matchup-based probabilities
- Provides detailed analytics on win probabilities and key performance factors
- Identifies tactical insights through multiple simulation runs

## Project Structure

```
t20_simulator/
├── data/                  # Data files
│   ├── teams/            # Team composition data
│   ├── players/          # Player statistics
│   ├── matches/          # Historical match data
│   └── venues/           # Venue statistics
├── output/               # Simulation outputs
│   ├── simulations/      # Raw simulation results
│   └── analysis/         # Analysis outputs
├── src/                  # Source code
│   ├── models/           # Core models (Player, Team, Match)
│   ├── simulation/       # Simulation engine and aggregator
│   ├── utils/            # Utility functions
│   └── visualization/    # Data visualization tools
└── tests/                # Test cases
```

## Key Components

### Data Models

1. **Player Model**: Implements detailed player statistics with phase-specific and matchup-based probabilities.
   - Represents the core simulation entity for each player
   - Handles probability calculations for different match situations
   - Models both batting and bowling capabilities

2. **Team Model**: Manages player selection and strategic decision-making.
   - Implements batting order creation and bowling rotation planning
   - Provides specialized selection for different match phases
   - Adjusts strategy based on match situation

3. **Match Model**: Handles the simulation of a complete T20 cricket match.
   - Simulates toss, innings, and individual ball outcomes
   - Tracks comprehensive match statistics
   - Identifies key moments and win probability progression

### Simulation Engine

1. **SimulationEngine**: Orchestrates efficient multi-simulation processing.
   - Runs multiple simulations in parallel for performance
   - Manages data loading and preparation
   - Tracks performance metrics

2. **ResultAggregator**: Analyzes simulation results to extract insights.
   - Calculates win probabilities with statistical confidence
   - Identifies key performance factors
   - Generates strategic insights

## Data Sources

The simulator uses several data files:

1. `squad_profiles.json`: Team compositions and player roles
2. `batting_stats.json`: Comprehensive batting statistics
3. `bowling_stats.json`: Comprehensive bowling statistics
4. `t20_bbb.csv`: Ball-by-ball data from historical matches

### Data Format Examples

#### batting_stats.json structure:
```json
{
  "<player_name>": {
    "bat_hand": "right",
    "batter_id": "p123",
    "total_runs": 1200,
    "by_phase": {
      "1": {
        "runs": 450,
        "balls": 320,
        ...
      }
    },
    "vs_bowler_styles": {
      "fast": {
        "runs": 600,
        "balls": 430,
        ...
      }
    },
    "vs_line_length": {
      "(good, fullish)": {
        "runs": 320,
        "balls": 180,
        ...
      }
    }
  }
}
```

## Usage

### Running a Single Match Simulation

```bash
python main.py --team1 IND --team2 AUS --venue MCG --mode single
```

### Running Batch Simulations

```bash
python main.py --team1 IND --team2 AUS --venue MCG --simulations 1000
```

### Command Line Arguments

- `--team1`: ID of the first team (required)
- `--team2`: ID of the second team (required)
- `--venue`: Optional venue ID
- `--simulations`: Number of simulations to run (default: 1000)
- `--mode`: Run a single match or batch simulation (`single` or `batch`)
- `--output`: Output directory for results (default: `output`)

## Implementation Notes

### Statistical Models

The simulator employs several statistical models:

1. **Phase-Based Batting Model**: Different probability distributions for powerplay, middle overs, and death overs.
2. **Matchup-Based Bowling Model**: Adjusts effectiveness against specific batsman types.
3. **Match Progression Model**: Handles momentum shifts and strategic adjustments.

### Optimization Techniques

For efficient simulation performance:

1. **Parallel Processing**: Uses multiprocessing to run simulations concurrently.
2. **Optimized Data Structures**: Employs preprocessed data for faster lookups.
3. **Batched Simulations**: Groups simulations for better CPU utilization.

## Analysis Outputs

The simulator produces several outputs:

1. **Raw Match Results**: Detailed ball-by-ball data for each simulation.
2. **Win Probability Analysis**: Statistical analysis of win probabilities.
3. **Key Factors Analysis**: Identification of decisive performance factors.
4. **Strategic Insights**: Actionable recommendations based on simulations.

## Future Enhancements

Planned improvements for the simulator:

1. Web interface for interactive simulation control
2. Real-time API integration for updated player statistics
3. Enhanced visualization of simulation results
4. More detailed player fatigue and form modeling
5. Support for tournament and series simulations

## Dependencies

- Python 3.8+
- NumPy
- Pandas
- Multiprocessing
- JSON

## License

MIT License
