import torch
import pufferlib.pytorch
import numpy as np
import os
import glob
import ast
import configparser
from collections import defaultdict

rep_size = 128


def puffer_type(value):
    try:
        return ast.literal_eval(value)
    except:
        return value


def load_config_for_probe(env_name: str) -> dict:
    import pufferlib

    puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')

    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split():
                break
        else:
            raise pufferlib.APIUsageError(f'No config for env_name {env_name}')

    args = defaultdict(dict)
    for section in p.sections():
        for key in p[section]:
            value = puffer_type(p[section][key])
            if section == 'base':
                args[key] = value
            else:
                if section not in args:
                    args[section] = {}
                args[section][key] = value

    args['load_model_path'] = None
    args['load_probe_path'] = None
    args['load_id'] = None
    args['render_mode'] = 'auto'
    args['train']['use_rnn'] = args.get('rnn_name') is not None

    return args


def add_config_args_to_parser(parser, env_name: str):
    import pufferlib

    puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    puffer_config_dir = os.path.join(puffer_dir, 'config/**/*.ini')
    puffer_default_config = os.path.join(puffer_dir, 'config/default.ini')

    if env_name == 'default':
        p = configparser.ConfigParser()
        p.read(puffer_default_config)
    else:
        for path in glob.glob(puffer_config_dir, recursive=True):
            p = configparser.ConfigParser()
            p.read([puffer_default_config, path])
            if env_name in p['base']['env_name'].split():
                break
        else:
            p = configparser.ConfigParser()
            p.read(puffer_default_config)

    for section in p.sections():
        for key in p[section]:
            fmt = f'--{key}' if section == 'base' else f'--{section}.{key}'
            parser.add_argument(
                fmt.replace('_', '-'),
                default=None,
                type=puffer_type,
                help=f'Override {section}.{key} from config'
            )


def apply_config_overrides(config: dict, parsed_args) -> dict:
    parsed = vars(parsed_args)
    for key, value in parsed.items():
        if value is None:
            continue
        if key in ('command', 'load_model_path', 'env', 'samples', 'device'):
            continue

        parts = key.split('.')
        if len(parts) == 1:
            config[key] = value
        else:
            section, subkey = parts[0], '.'.join(parts[1:])
            if section not in config:
                config[section] = {}
            config[section][subkey] = value

    return config


def load_env_for_probe(env_name: str, args: dict):
    from pufferlib.pufferl import load_env
    return load_env(env_name, args)


def load_policy_for_probe(args: dict, vecenv, env_name: str):
    from pufferlib.pufferl import load_policy
    return load_policy(args, vecenv, env_name)


class Probe(torch.nn.Module):
    def __init__(self, hidden_dim: int = 32, num_feature_types: int = 3, obs_size: int = 7):
        super().__init__()
        self.num_feature_types = num_feature_types
        self.obs_size = obs_size
        self.num_classes = obs_size * obs_size + 1

        self.probes = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, self.num_classes)
            for _ in range(num_feature_types)
        ])

    def train(self, data_embeddings, labels, num_epochs=10, learning_rate=0.001, batch_size=32,
              val_embeddings=None, val_labels=None, early_stopping=True, patience=10,
              min_delta=0.0, save_path='experiments/probe/probe.pt'):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=learning_rate)

        def compute_total_loss(eval_embeddings, eval_labels):
            total_eval_loss = 0.0
            total_eval_batches = 0
            with torch.no_grad():
                for j in range(0, len(eval_embeddings), batch_size):
                    eb = eval_embeddings[j:j+batch_size]
                    el = eval_labels[j:j+batch_size].long()
                    batch_loss = 0.0
                    for feature_idx in range(self.num_feature_types):
                        batch_loss += criterion(self.probes[feature_idx](eb), el[:, feature_idx])
                    total_eval_loss += batch_loss.item()
                    total_eval_batches += 1
            return total_eval_loss / max(total_eval_batches, 1)

        have_val = (val_embeddings is not None) and (val_labels is not None)
        use_early_stopping = early_stopping and have_val
        best_val_loss = float('inf')
        epochs_no_improve = 0
        saved_once = False

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size].detach()
                batch_labels = labels[i:i+batch_size].long()

                total_loss = 0.0
                for feature_idx in range(self.num_feature_types):
                    total_loss += criterion(self.probes[feature_idx](batch_embeddings), batch_labels[:, feature_idx])

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)

            if have_val:
                val_loss = compute_total_loss(val_embeddings, val_labels)
                if (best_val_loss - val_loss) > min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if save_path:
                        dir_name = os.path.dirname(save_path)
                        if dir_name:
                            os.makedirs(dir_name, exist_ok=True)
                        torch.save(self.state_dict(), save_path)
                        saved_once = True
                else:
                    epochs_no_improve += 1

            if use_early_stopping and epochs_no_improve >= patience:
                break

        if save_path and not saved_once:
            dir_name = os.path.dirname(save_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            torch.save(self.state_dict(), save_path)
            saved_once = True

        if have_val and saved_once and save_path:
            try:
                self.load_state_dict(torch.load(save_path, map_location='cpu'))
            except Exception as e:
                print(f'Warning: failed to reload best model from {save_path}: {e}')

    def evaluate(self, data_embeddings, labels, batch_size=32):
        for probe in self.probes:
            probe.eval()

        all_predicted = []
        all_labels = []
        all_logits = [[] for _ in range(self.num_feature_types)]

        with torch.no_grad():
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size]
                batch_labels = labels[i:i+batch_size].long()

                batch_predictions = []
                for feature_idx in range(self.num_feature_types):
                    logits = self.probes[feature_idx](batch_embeddings)
                    batch_predictions.append(logits.argmax(dim=1).unsqueeze(1))
                    all_logits[feature_idx].append(logits)

                all_predicted.append(torch.cat(batch_predictions, dim=1))
                all_labels.append(batch_labels)

        all_predicted = torch.cat(all_predicted, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        per_feature_accuracy = []
        per_feature_cross_entropy = []
        criterion = torch.nn.CrossEntropyLoss()

        for feature_idx in range(self.num_feature_types):
            feature_logits = torch.cat(all_logits[feature_idx], dim=0)
            acc = (all_predicted[:, feature_idx] == all_labels[:, feature_idx]).float().mean().item()
            per_feature_accuracy.append(acc)
            per_feature_cross_entropy.append(criterion(feature_logits, all_labels[:, feature_idx]).item())

        overall_accuracy = (all_predicted == all_labels).float().mean().item()
        mean_cross_entropy = sum(per_feature_cross_entropy) / len(per_feature_cross_entropy)

        return {
            'overall_accuracy': overall_accuracy,
            'mean_cross_entropy': mean_cross_entropy,
            'per_feature_accuracy': per_feature_accuracy,
            'per_feature_cross_entropy': per_feature_cross_entropy,
        }


def collect_probe_dataset(env_name, vecenv=None, policy=None, args=None, num_samples=50, device='cpu', feature_source='representation'):
    args = args or load_config_for_probe(env_name)
    args['vec'] = dict(backend='Serial', num_envs=1)
    args['env']['random_sampling'] = True

    vecenv = vecenv or load_env_for_probe(env_name, args)
    policy = policy or load_policy_for_probe(args, vecenv, env_name)

    ob, info = vecenv.reset()
    driver = vecenv.driver_env
    num_agents = vecenv.observation_space.shape[0]

    state = {}
    if args['train']['use_rnn']:
        state = dict(
            lstm_h=torch.zeros(num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(num_agents, policy.hidden_size, device=device),
        )

    policy.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            ob_tensor = torch.as_tensor(ob).to(device)

            if feature_source == 'observation':
                features = ob_tensor
            else:
                features = policy.policy.encode_observations(ob_tensor)

            counts = driver.get_ground_truth_counts()
            ground_truth = torch.tensor(counts, dtype=torch.long)

            embeddings_list.append(features.cpu())
            labels_list.append(ground_truth)

            logits, value = policy.forward_eval(ob_tensor, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)

            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)

            ob = vecenv.step(action)[0]

    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return embeddings, labels


def train_probe(embeddings, labels, hidden_dim=rep_size, num_feature_types=3, obs_size=7,
                num_epochs=50, learning_rate=0.001, train_split=0.8,
                early_stopping=True, patience=10, min_delta=0.0,
                save_path='experiments/probe/probe.pt'):
    if hidden_dim is None:
        hidden_dim = embeddings.shape[1]

    probe = Probe(hidden_dim=hidden_dim, num_feature_types=num_feature_types, obs_size=obs_size)

    n_samples = len(embeddings)
    n_train = int(n_samples * train_split)
    indices = torch.randperm(n_samples)

    train_embeddings = embeddings[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_embeddings = embeddings[indices[n_train:]]
    val_labels = labels[indices[n_train:]]

    probe.train(
        train_embeddings, train_labels,
        num_epochs=num_epochs, learning_rate=learning_rate,
        val_embeddings=val_embeddings, val_labels=val_labels,
        early_stopping=early_stopping, patience=patience, min_delta=min_delta,
        save_path=save_path,
    )

    if len(val_embeddings) > 0:
        metrics = probe.evaluate(val_embeddings, val_labels)
    else:
        metrics = {"overall_accuracy": 0.0, "mean_cross_entropy": 0.0}

    return probe, metrics


def evaluate_probe(probe, env_name, vecenv=None, policy=None, args=None, num_samples=50, device='cpu', feature_source='representation'):
    embeddings, labels = collect_probe_dataset(env_name, vecenv, policy, args, num_samples, device, feature_source)
    return probe.evaluate(embeddings, labels)


def update_probe(env_name, policy, accuracy, checkpoint, existing_probe=None, hidden_dim=rep_size, obs_size=7, feature_source='representation', num_feature_types=3):
    args = load_config_for_probe(env_name)
    embeddings, labels = collect_probe_dataset(env_name, args=args, policy=policy, num_samples=10, feature_source=feature_source)
    labels = labels[:, :num_feature_types]

    num_epochs = 5
    if existing_probe is None:
        probe = Probe(hidden_dim=hidden_dim, num_feature_types=num_feature_types, obs_size=obs_size)
        num_epochs = 10
    else:
        probe = existing_probe

    n_samples = len(embeddings)
    n_train = int(n_samples * 0.8)
    indices = torch.randperm(n_samples)
    train_embeddings = embeddings[indices[:n_train]]
    train_labels = labels[indices[:n_train]]
    val_embeddings = embeddings[indices[n_train:]]
    val_labels = labels[indices[n_train:]]

    probe.train(
        train_embeddings, train_labels,
        num_epochs=num_epochs, learning_rate=0.001,
        val_embeddings=val_embeddings, val_labels=val_labels,
        early_stopping=False, save_path=None,
    )

    if len(val_embeddings) > 0:
        metrics = probe.evaluate(val_embeddings, val_labels)
        accuracy[checkpoint] = metrics

    return probe


def resolve_model_path(load_model_path, env_name):
    if load_model_path == 'latest':
        candidates = glob.glob(f"experiments/{env_name}*.pt")
        if not candidates:
            raise FileNotFoundError(f"No checkpoints found matching experiments/{env_name}*.pt")
        resolved = max(candidates, key=os.path.getctime)
        print(f"Resolved 'latest' to: {resolved}")
        return resolved
    return load_model_path


if __name__ == "__main__":
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("command", nargs="?", default=None)
    pre_parser.add_argument("--env", type=str, default="puffer_repgrid")
    pre_args, remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Linear probing tools for PufferLib",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pufferlib.probing count --load-model-path experiments/model.pt
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    count_parser = subparsers.add_parser("count", help="Count-based probe test")
    count_parser.add_argument("--load-model-path", type=str, default=None)
    count_parser.add_argument("--env", type=str, default="puffer_repgrid")
    count_parser.add_argument("--samples", type=int, default=50)
    add_config_args_to_parser(count_parser, pre_args.env)

    args = parser.parse_args()

    if args.command == "count":
        config = load_config_for_probe(args.env)
        config = apply_config_overrides(config, args)
        config['vec'] = dict(backend='Serial', num_envs=1)
        device = config['train'].get('device', 'cpu')
        vecenv = load_env_for_probe(args.env, config)
        policy = load_policy_for_probe(config, vecenv, args.env)
        if args.load_model_path:
            model_path = resolve_model_path(args.load_model_path, args.env)
            print(f"Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            policy.load_state_dict(state_dict)
        policy.to(device)
        embeddings, labels = collect_probe_dataset(
            args.env, args=config, policy=policy, num_samples=args.samples, device=device
        )
        probe, metrics = train_probe(embeddings, labels)
        evaluate_probe(probe, args.env, policy=policy, args=config, device=device)
    else:
        parser.print_help()
