# LensRL

## Reinforcement Learning for Lens Design

**LensRL** is a scientific toolkit designed to apply reinforcement learning (RL) techniques to lens design. By leveraging the optical analysis capabilities of [Optiland](https://github.com/HarrisonKramer/optiland), this repository provides a modular and extensible framework to explore and optimize lens systems using RL methodologies.

> [!NOTE]
> This repository is under active development, and its API may evolve as new features are introduced.

## What is LensRL?

LensRL is a modular platform that recasts lens design as an RL problem. It provides a suite of components that enable the design and optimization of optical systems through a systematic RL approach. Key functionalities include:

- **Action Module:** A dynamic set of actions to adjust lens parameters.
- **Reward Functions:** Customizable rewards based on RMS spot size, system complexity, aperture size, field of view, etc., guiding the RL agent towards optimal designs.
- **Observation & Action Spaces:** RL spaces that encapsulate the state of the optical system and drive decision-making.
- **Configurable Optical System:** A flexible class that integrates with [Optiland](https://github.com/HarrisonKramer/optiland) for detailed optical simulations and analysis.
- **Normalization Module:** Standardizes variables to improve learning stability.
- **Lens Design Environment:** A dedicated environment that frames lens design challenges as RL tasks, facilitating automated exploration and iterative improvement.

## Why LensRL?

Lens design inherently involves balancing multiple objectives and constraints. By framing lens design as an RL problem, LensRL aims to:

- Automate and accelerate the optimization process.
- Navigate complex design spaces more efficiently.
- Enable researchers, engineers, or enthusiasts to experiment with different RL strategies to achieve innovative optical designs.

## Getting Started

To get started, follow these steps:

1. **Clone the LensRL Repository:**

```sh
git clone https://github.com/HarrisonKramer/LensRL.git
```

2. **Install LensRL dependencies**:

```sh
cd LensRL
pip install -r requirements.txt
```

3. **Customize and Experiment:** Modify reward functions, tweak action spaces, and tailor the environment to meet your research objectives.

## Examples

[A minimal working example can be found here](https://github.com/HarrisonKramer/LensRL/examples/minimum_working_example.ipynb). More notebooks will be added as development continues.

## Contributing

If you have feedback, would like to contribute, or have any ideas for how to make LensRL better, feel free to open an issue or submit a pull request.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
