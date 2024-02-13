# MEV-Boost Auction Simulation Framework

<img src="https://ethresear.ch/uploads/default/original/2X/8/8b838cb489dea1a8bbb4c093ccc0fee1e91fff0e.jpeg" alt="img" width="700"/>


Developed by [Fei](https://twitter.com/William33203632) and [soispoke.eth](https://twitter.com/soispoke).

For details about the Game-theoretic MEV-Boost Auction Model, users are strongly recommended to check the [MMA post](https://ethresear.ch/t/game-theoretic-model-for-mev-boost-auctions-mma/16206) and [bidding war paper](https://arxiv.org/abs/2312.14510).

The simulation is built using the [Mesa framework](https://mesa.readthedocs.io/en/stable/) for agent-based modeling in Python. 

## Features
Auction simulation: Bids submitted by agents with different strategies per time step in the MEV-Boost auction .

Multiple bidding strategies: Naive, Adaptive, Last Minute, Stealth, and Bluff strategies.

Data Collection and Visualization: Collects bidding data throughout the simulation and visualizes the results.

## Setup
To run this simulation, you will need Python 3.8 (or above) installed on your system along with the following Python libraries:

Mesa, NumPy, Pandas, Matplotlib, SciPy


## Components
The **Player class** defines and initializes basic parameters shared by all types of agents.

Each **PlayerWithStrategy class** is a subclass of the **Player class** that defines the bidding behaviors of each type of agent using the corresponding strategy.

The **Auction class** manages the simulation environment, including signal updates, agent interactions, and data collection. 

## Data Collection and Visualization
Data collection is achieved by Mesa.Datacollector(). The simulation collects data on bids, agent strategies, and signal values at each step. After the simulation ends, it visualizes the bidding dynamics over time and displays the final auction outcomes.

## Parameter Values
For parameter value settings, some useful resources are listed here:

[BBP Post](https://ethresear.ch/t/empirical-analysis-of-builders-behavioral-profiles-bbps/16327)

[mevboost.pics](https://mevboost.pics/)

[orderflow.art](https://orderflow.art/)


## Customization
You can customize the simulation by adjusting the number of agents using each strategy, agent parameters when adding agents into the auction model, and the auction parameters in the Auction class initialization within the script. Or, create new strategies and use the framework for simulation.

## License
This project is open-source and available under the MIT License.
