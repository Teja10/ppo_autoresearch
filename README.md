# ppo_autoresearch

## Canonical Task Decision

The first autoresearch loop uses exactly one PPO robotics task:
`FetchPushDense-v3` from Gymnasium-Robotics.

`FetchPushDense-v3` is the canonical task because it is available through the
Gymnasium-Robotics MuJoCo simulator stack, runs locally on OSX for cheap probe
experiments, and still represents object manipulation that should require
larger paid Modal runs to evaluate PPO variants across bigger budgets and
seeds.

`FetchReach` may be used only as an install or simulator smoke check. It is not
the research task. Multi-task benchmarking is out of scope for the first loop.

### Simulator Contract

- Simulator package set: `gymnasium==1.3.0`,
  `gymnasium-robotics==1.3.1`, `mujoco==3.1.6`.
- Environment id: `FetchPushDense-v3`.
- Simulator backend: MuJoCo through Gymnasium-Robotics.
- Reward type: dense Fetch Push reward.
- Observation space: dict with `achieved_goal` shape `(3,)`, `desired_goal`
  shape `(3,)`, and `observation` shape `(25,)`.
- Action space: continuous `Box(-1.0, 1.0, (4,), float32)`.

### Local Verification

Verified locally on OSX with the pinned simulator package set:

```bash
uv run \
  --with gymnasium==1.3.0 \
  --with gymnasium-robotics==1.3.1 \
  --with mujoco==3.1.6 \
  python -c 'import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)
env = gym.make("FetchPushDense-v3")
env.reset(seed=0)
spaces = env.observation_space.spaces
print(f"id={env.spec.id}")
print(f"achieved_goal_shape={spaces['achieved_goal'].shape}")
print(f"desired_goal_shape={spaces['desired_goal'].shape}")
print(f"observation_shape={spaces['observation'].shape}")
print(f"action_space={env.action_space}")
print(f"obs_keys={sorted(spaces.keys())}")
env.close()'
```

Observed verification output:

```text
id=FetchPushDense-v3
achieved_goal_shape=(3,)
desired_goal_shape=(3,)
observation_shape=(25,)
action_space=Box(-1.0, 1.0, (4,), float32)
obs_keys=['achieved_goal', 'desired_goal', 'observation']
```
