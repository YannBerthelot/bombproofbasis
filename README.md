# bombproofbasis

Bombproofbasis is a library that provides implementation of Reinforcement Learning Agents using PyTorch. It came from the need for an A2C agent with LSTM capability (not available in stable-baselines3, which is the main inspiration for this package) as well as a simpler code for easier modifications.

It is designed for easy research work. Its design philosophy revolves around:

- Automatic logging of parameters (network architectures, hyperparameters)
- Automatic logging of results (Train cumulative reward, Test episode reward)
- Automatic logging of debugging KPIs (advantages distributions, value targets distribution, network weights ...)
- Possibility to log in Tensorboard and/or WandB
- Modular code for easy modification and creation of new agents.
- Providing an easy way to define neural network, including LSTMs.

This comes in addition to having a solid and robust package :

- Unit tests to ensure non-regression
- Probe environments to help debugging RL algorithms (as described in : https://andyljones.com/posts/rl-debugging.html)
- Fixed types and dataclasses to ensure the right types.

## Algorithms available:

- A2C : Monte-Carlo, Temporal-Difference, 2-steps with adjustable batch size.

## Installation

This package uses poetry if you wish to build it yourself.

```bash
git clone https://github.com/YannBerthelot/bombproofbasis.git
cd bombproofbasis
poetry install
```

Not yet available on PyPi.

## Usage

To run the examples:

```bash
python -m bombproofbasis
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## To-do

-[X] Add seeding of everything for reproducibility (for research and debugging)
-[ ] Extend probe environments for MC and n-steps
-[ ] Benchmark agents
-[ ] Find a way to check if LSTMs add anything
-[ ] Extend from 2-steps to n-steps
-[ ] Extend batch-size for n-steps
-[ ] Add parallel-env support
