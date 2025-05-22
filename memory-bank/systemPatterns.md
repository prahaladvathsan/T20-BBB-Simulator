# System Patterns

## System Architecture
The T20-BBB-Simulator is built around a modular architecture with clear separation of concerns:

1. **Core Modules Layer** - Contains the main simulation components (Engine, Team, Player, Match)
2. **Data Layer** - Handles loading and processing of cricket statistics
3. **Simulation Layer** - Manages execution of simulations and parallel processing
4. **Analysis Layer** - Processes simulation results and calculates probabilities
5. **Integration Layer** - (Planned) For batting impact and DLS method integration

## Key Technical Decisions
- Modular class-based design for core simulation components
- Event-driven approach for ball-by-ball simulation
- Multi-process execution for efficient parallel simulations
- Synthetic data generation for initial testing
- Statistical modeling for win probability calculations
- Integration planning for batting impact and DLS method

## Design Patterns
- **Factory Pattern** - For creating player and team objects from data
- **Strategy Pattern** - For different batting/bowling strategies
- **Observer Pattern** - For tracking match state and ball-by-ball events
- **Composite Pattern** - For team composition and role management
- **Command Pattern** - For simulation execution and batch processing

## Component Relationships
```
DataLoader --> [Engine, Team, Player, Match] --> SimulationEngine --> ResultAnalyzer
```

- **DataLoader** provides data to create core objects
- **Engine** manages the simulation process
- **Team** and **Player** objects handle team and player logic
- **Match** objects coordinate the actual simulation
- **ResultAnalyzer** processes simulation outputs

## Data Flow
- **Input**: 
  1. Synthetic data (current)
  2. Real cricket datasets (in progress)
- **Processing**: 
  1. Data loading and preprocessing
  2. Core object creation
  3. Match simulation with ball-by-ball events
  4. Parallel execution of simulations
  5. Result analysis and probability calculations
- **Output**: 
  1. Match results
  2. Win probabilities
  3. Outcome predictions
  4. (Planned) Batting impact analysis
  5. (Planned) DLS calculations

## Core Modules
1. **Engine** - Main simulation controller
2. **Team** - Team composition and strategy
3. **Player** - Player statistics and performance
4. **Match** - Match simulation and state tracking
5. **DataLoader** - Data processing and validation
6. **ResultAnalyzer** - (Planned) Advanced analysis and predictions

## APIs and Interfaces
- **Engine** - Interface for simulation control
- **Team** - API for team management
- **Player** - Interface for player statistics
- **Match** - API for match simulation
- **DataLoader** - Interface for data processing
- **ResultAnalyzer** - (Planned) API for advanced analysis

## Testing Approach
- Unit testing of core modules
- Integration testing of module interactions
- Advanced testing with league-wide simulations
- Validation against synthetic data
- (Planned) Validation against real match data 