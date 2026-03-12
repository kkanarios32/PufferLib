import gymnasium
import numpy as np
import os

import pufferlib
from pufferlib.ocean.repgrid import binding

SAMPLING_MODE = "spiral"

FEATURE_DIMS = {
    "mnist": 49,
    "spiral": 3,
    "gaussian": 3,
}

_MNIST_CACHE = None

def load_mnist_data():
    """Load FashionMNIST classes, downsample to 7x7, normalize.
    Returns dict {0..4} -> (N, 49) arrays.
    FashionMNIST mapping: T-shirt=dogs, Pullover=cats, Coat=tigers, Shirt=empty, Dress=keys.
    """
    global _MNIST_CACHE
    if _MNIST_CACHE is not None:
        return _MNIST_CACHE

    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((7, 7)),
        transforms.ToTensor(),
    ])

    cache_dir = os.path.join(os.path.dirname(__file__), '.mnist_cache')
    fmnist = datasets.FashionMNIST(root=cache_dir, train=True, download=True, transform=transform)

    class_map = {0: 0, 2: 1, 4: 2, 6: 3, 3: 4}
    data_by_class = {k: [] for k in range(5)}

    for img, label in fmnist:
        if label in class_map:
            data_by_class[class_map[label]].append(img.numpy().flatten())

    for k in data_by_class:
        data_by_class[k] = np.array(data_by_class[k], dtype=np.float32)

    all_data = np.concatenate([data_by_class[k] for k in range(5)], axis=0)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0) + 1e-8
    for k in data_by_class:
        data_by_class[k] = (data_by_class[k] - global_mean) / global_std

    _MNIST_CACHE = data_by_class
    return data_by_class


class RepGrid(pufferlib.PufferEnv):
    def __init__(self, num_envs=1, render_mode=None, log_interval=128,
                 size=7, buf=None, seed=0, random_sampling=False, key_door=False):
        feature_dim = FEATURE_DIMS[SAMPLING_MODE]

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(size * size * (1 + feature_dim),), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(4)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.random_sampling = random_sampling
        self.size = size
        self.feature_dim = feature_dim

        super().__init__(buf)

        init_kwargs = dict(
            size=size, random_sampling=random_sampling, key_door=key_door,
            feature_dim=feature_dim, sampling_mode=SAMPLING_MODE,
        )
        if SAMPLING_MODE == "mnist":
            mnist_data = load_mnist_data()
            for i in range(5):
                init_kwargs[f'mnist_{i}'] = mnist_data[i]

        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations, num_envs, seed, **init_kwargs,
        )

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.log_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)

    def get_ground_truth_counts(self):
        return binding.vec_get_counts(self.c_envs)

    def set_probe_counts(self, env_id, dogs, cats, tigers, keys=0):
        binding.vec_set_probe_counts(self.c_envs, int(env_id), int(dogs), int(cats), int(tigers), int(keys))

if __name__ == "__main__":
    import time

    N = 4096
    env = RepGrid(num_envs=N)
    env.reset()
    steps = 0

    CACHE = 1024
    actions = np.random.randint(0, 5, (CACHE, N))

    i = 0
    start = time.time()
    while time.time() - start < 10:
        env.step(actions[i % CACHE])
        steps += N
        i += 1

    print("RepGrid SPS:", int(steps / (time.time() - start)))
