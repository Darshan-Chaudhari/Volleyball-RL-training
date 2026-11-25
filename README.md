# Volleyball-RL-training
2D volleyball in Pygame where two AI agents learn to rally through self-play, using a PettingZoo environment and Ray RLlib PPO.
<img width="552" height="263" alt="Screenshot 2026-11-25 at 1 15 41 PM" src="https://github.com/user-attachments/assets/dfcb0623-be4f-43a8-a550-ddd4373f1792" />
# Volleyball RL

A 2D side-view volleyball game, written from scratch in Pygame, where two AI players learn to rally through self-play. The game is wrapped as a [PettingZoo](https://pettingzoo.farama.org/) environment and trained with PPO in [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html), using one neural network shared by both players.

**Video walkthrough:** (https://youtu.be/uGXnX8b3dbM)

<!-- Add a GIF or screenshot of the trained agents here, e.g. ![demo](media/demo.gif) -->

## What it does

- Custom physics: gravity, bouncing ball, net collision, jumping players
- Hit direction depends on where the ball meets the player, and a player above the ball hits it downward, so spikes are possible
- Volleyball rules: three touches per side, touch counter resets when the ball crosses the net, faults for dropped balls, out of bounds, a fourth touch, and a missed serve
- A PettingZoo `ParallelEnv` with a 16-value observation and 5 discrete actions
- PPO self-play training with TensorBoard logging
- A viewer that loads any checkpoint and renders the agents playing

## Results

All numbers come from the training graphs logged during these runs, not from a separate benchmark.

| Run | Training | Average rally length | Average episode return |
|---|---|---|---|
| Run 1 | about 4 hours, from scratch | about 100 steps to about 190 steps | about 0.06 to about 0.14 |
| Run 2 | about 1.3 hours, continued from the best Run 1 checkpoint, with two added net rewards | peaked above 220, settled near 190-200 | about 0.42 |

How to read this:

- Return is not comparable across the two runs, because Run 2 added a +0.2 net-crossing bonus and a -0.05 net-hit penalty. Part of its higher return is that reward.
- In self-play, one player's win is the other's loss, so the win/lose rewards roughly cancel in the summed return. Return mostly reflects ball contacts and net crossings, and rally length is the more informative curve.
- Run 2 was a continuation, not an independent run, and the two reward changes were made together, so this project cannot say which one helped.

## Limitations

This is a first working version, not polished yet.

- The agents track the ball, receive serves, and send it over the net, but they do not reliably set up pass and spike plays.
- In Run 1, about half of the hits sent the ball into the net or were bumped twice on the same side. Run 2 improved this but did not eliminate it.
- Action 4 ("hit") currently has no effect. Hits happen automatically on contact, so the action exists but does nothing.
- The environment is 1v1. The observation reserves four zero-filled slots for a teammate and a second opponent, so 2v2 is possible without changing the input size.
- Training was run on a laptop CPU for a few hours in total, so there is no evidence about what longer training would do.

## How it works

### Environment (`volleyball_env.py`)

One rally is one episode. An episode ends when a fault happens (termination) or after 1000 steps (truncation).

**Observation (16 values, normalized):**

| Values | Meaning |
|---|---|
| 4 | Own x, y, vx, vy |
| 2 | Teammate x, y (zeros in 1v1) |
| 4 | Ball x, y, vx, vy |
| 2 | Opponent x, y |
| 2 | Second opponent x, y (zeros in 1v1) |
| 1 | Touches used by the current side, divided by the 3-touch limit |
| 1 | Whether the player is in the air |

**Actions (`Discrete(5)`):** 0 do nothing, 1 left, 2 right, 3 jump, 4 hit (no effect yet).

**Rewards:**

| Event | Reward |
|---|---|
| Win a rally | +1 |
| Lose a rally (including a missed serve) | -1 |
| Touch the ball | +0.05 |
| Send the ball over the net | +0.2 |
| Ball hits the net | -0.05 |

The net-crossing bonus and net-hit penalty were added for Run 2.

### Training (`train.py`)

PPO in RLlib with a single shared policy for both players. Settings at the top of the file:

- 4 environment runners, a time budget in hours, and a checkpoint every 10 iterations
- Learning rate 3e-4, gamma 0.99, batch size 8000, minibatch size 512, 10 epochs, entropy coefficient 0.01
- `RESTORE_PATH` can point to an earlier checkpoint to continue training from it

Training logs `episode_len_mean` and `episode_return_mean` to TensorBoard.

## Project layout

| File | Purpose |
|---|---|
| `constants.py` | All physics, court, and reward constants in one place |
| `entities.py` | `Ball`, `Player`, and the hit-detection function |
| `volleyball_env.py` | The PettingZoo environment |
| `main.py` | Two-player keyboard version of the game |
| `test_env.py` | Runs the environment with random actions and renders it |
| `train.py` | The main training script |
| `play_checkpoint.py` | Loads a checkpoint and renders the agents playing |

## Setup

Python 3.11 is recommended. Pygame and Ray did not install cleanly on Python 3.14 when this was built.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install pygame pettingzoo gymnasium "ray[rllib]" torch tensorboard numpy
```

Versions this was developed with: Python 3.11.9, pygame 2.6.1, Ray [FILL IN: run `pip show ray`]. RLlib's API changes between versions, so if training or checkpoint loading fails, check your Ray version first. A `requirements.txt` with exact versions is included.

## Usage

Play it yourself (left player: A / D / W, right player: arrow keys):

```bash
python main.py
```

Watch random agents (a quick check that the environment works):

```bash
python test_env.py
```

Train:

```bash
python train.py
```

Watch the training curves:

```bash
tensorboard --logdir tb_logs
```

Watch a trained checkpoint (add `--greedy` for deterministic actions):

```bash
python play_checkpoint.py checkpoints/iter_00100
```

Checkpoints are large and are not included in the repository. [FILL IN: link to a downloadable checkpoint, or remove this line.]

## Ideas for next steps

- 2v2 with four agents, using the reserved observation slots
- A separate hit action so spikes are a decision instead of a side effect of contact
- Log each reward component separately to see which ones help
- Compare the two Run 2 changes (net bonus and net penalty) in separate runs
- Self-play against a pool of past versions to avoid stalling
- A human-vs-AI mode

## License
- NA for now

