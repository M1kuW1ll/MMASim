# MEV-Boost Auction Simulation Framework

Developed by [Fei](https://twitter.com/William33203632) and [soispoke.eth](https://twitter.com/soispoke).

For details about the Game-theoretic MEV-Boost Auction Model, please refer to the [MMA post](https://ethresear.ch/t/game-theoretic-model-for-mev-boost-auctions-mma/16206) and [bidding war paper](https://arxiv.org/abs/2312.14510).

The simulation is built using the [Mesa framework](https://mesa.readthedocs.io/en/stable/) for agent-based modeling in Python. 

## Features
Simulation: Bids by agents with different strategies in MEV-Boost Auction per time step.

Multiple bidding strategies: Naive, Adaptive, Last Minute, Stealth, and Bluff strategies.

Data Collection and Visualization: Collects bidding data throughout the simulation and visualizes the results.

## Setup
To run this simulation, you will need Python 3.8 (or above) installed on your system along with the following Python libraries:

Mesa, NumPy, Pandas, Matplotlib, SciPy


## Components
The **Player class** defines and initializes basic parameters shared by all types of agents.

Each **PlayerWithStrategy class** is a subclass of the **Player class** that defines the bidding behaviors of each type of agent using corresponding strategies.

The **Auction class** manages the simulation environment, including signal updates, agent interactions, and data collection. 

## Data Collection and Visualization
Data collection is achieved by Mesa.Datacollector(). The simulation collects data on bids, agent strategies, and signal values at each step. After the simulation ends, it visualizes the bidding dynamics over time and displays the final auction outcomes.

## Customization
You can customize the simulation by adjusting the number of agents using each strategy, agent parameters when adding agents into the auction model, and the auction parameters in the Auction class initialization within the script. Or, create new strategies and use the framework for simulation.

## License
This project is open-source and available under the MIT License.
