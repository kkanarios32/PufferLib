import torch
import logging
from pufferlib.pufferl import load_env, load_policy, load_config
import pufferlib.pytorch
import numpy as np
import os

rep_size = 16

class Probe(torch.nn.Module):
    def __init__(self, hidden_dim: int = 32, num_feature_types: int = 3, obs_size: int = 7)  -> None:
        '''
        Initialize the probe for count classification.
        :param hidden_dim: The dimensionality of the hidden layer of the probe.
        :param num_feature_types: The number of feature types to predict (dogs, cats, tigers).
        :param obs_size: The size of the observation grid (obs_size x obs_size total positions).
        :return: None
        '''
        super().__init__()
        self.num_feature_types = num_feature_types
        self.obs_size = obs_size
        self.num_classes = obs_size * obs_size + 1  # classes: 0, 1, 2, ..., obs_size^2 (max possible count)
        
        self.probes = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, self.num_classes)
            for _ in range(num_feature_types)
        ])


    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epochs: int = 10,
              learning_rate: float = 0.001, batch_size: int = 32,
              val_embeddings: torch.Tensor = None, val_labels: torch.Tensor = None,
              early_stopping: bool = True, patience: int = 10, min_delta: float = 0.0,
              save_path: str = 'experiments/probe/probe.pt') -> None:
        '''
        Train the probe on the embeddings of data from the model.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape (N, num_feature_types), where each element is the count of that animal type.
        :param num_epochs: The number of epochs to train the probe for. An epoch is one pass through the entire dataset.
        :param learning_rate: How fast the probe learns.
        :param batch_size: Used to batch the data for computational efficiency.
        :param val_embeddings: Optional validation embeddings for early stopping.
        :param val_labels: Optional validation labels for early stopping.
        :param early_stopping: Whether to use early stopping when validation data is provided.
        :param patience: Number of epochs without improvement to wait before stopping.
        :param min_delta: Minimum change to qualify as an improvement.
        :param save_path: File path to save the best model.
        :return: None
        '''

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.probes.parameters(), lr=learning_rate)

        print('Training the probe...')

        def compute_total_loss(eval_embeddings: torch.Tensor, eval_labels: torch.Tensor) -> float:
            total_eval_loss = 0.0
            total_eval_batches = 0
            with torch.no_grad():
                for j in range(0, len(eval_embeddings), batch_size):
                    eb = eval_embeddings[j:j+batch_size]
                    el = eval_labels[j:j+batch_size].long()
                    batch_loss = 0.0
                    for feature_idx in range(self.num_feature_types):
                        logits = self.probes[feature_idx](eb)
                        feature_labels = el[:, feature_idx]
                        batch_loss += criterion(logits, feature_labels)
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
                    logits = self.probes[feature_idx](batch_embeddings)
                    feature_labels = batch_labels[:, feature_idx]
                    loss = criterion(logits, feature_labels)
                    total_loss += loss

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

            if epoch > 0 and (epoch % 10 == 0 or epoch == num_epochs - 1):
                if have_val:
                    print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}')

            if use_early_stopping and epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).')
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

        print('Training complete.')


    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        '''
        Get the probe's predictions on the embeddings from the model, for unseen data.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A tensor of shape (N, num_feature_types), where each element is the predicted number of that feature type.
        '''

        for probe in self.probes:
            probe.eval()
        
        all_predicted = []

        with torch.no_grad():
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size]
                
                batch_predictions = []
                for feature_idx in range(self.num_feature_types):
                    logits = self.probes[feature_idx](batch_embeddings) # (batch_size, num_classes)
                    predictions = torch.argmax(logits, dim=1) # (batch_size,)
                    batch_predictions.append(predictions.unsqueeze(1))
                    
                batch_output = torch.cat(batch_predictions, dim=1) # (batch_size, num_feature_types)
                all_predicted.append(batch_output)

        return torch.cat(all_predicted, dim=0)


    def evaluate(self, data_embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int = 32) -> dict:
        '''
        Evaluate the probe's performance by testing it on unseen data.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape (N, num_feature_types), where each element is the true count of that animal type.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A dictionary containing evaluation metrics.
        '''

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
                    logits = self.probes[feature_idx](batch_embeddings) # (batch_size, num_classes)
                    predictions = torch.argmax(logits, dim=1) # (batch_size,)
                    batch_predictions.append(predictions.unsqueeze(1))
                    all_logits[feature_idx].append(logits)
                
                batch_output = torch.cat(batch_predictions, dim=1) # (batch_size, num_feature_types)
                all_predicted.append(batch_output)
                all_labels.append(batch_labels)

        all_predicted = torch.cat(all_predicted, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        per_feature_accuracy = []
        per_feature_cross_entropy = []
        criterion = torch.nn.CrossEntropyLoss()
        
        for feature_idx in range(self.num_feature_types):
            feature_logits = torch.cat(all_logits[feature_idx], dim=0)
            feature_predictions = all_predicted[:, feature_idx]
            feature_labels = all_labels[:, feature_idx]
            
            correct = (feature_predictions == feature_labels).float()
            accuracy = correct.mean().item()
            per_feature_accuracy.append(accuracy)
            
            ce_loss = criterion(feature_logits, feature_labels).item()
            per_feature_cross_entropy.append(ce_loss)

        # Overall accuracy
        overall_accuracy = (all_predicted == all_labels).float().mean().item()
        mean_cross_entropy = sum(per_feature_cross_entropy) / len(per_feature_cross_entropy)

        metrics = {
            'overall_accuracy': overall_accuracy,
            'mean_cross_entropy': mean_cross_entropy,
            'per_feature_accuracy': per_feature_accuracy,
            'per_feature_cross_entropy': per_feature_cross_entropy,
        }

        print(f'Probe Overall Accuracy: {overall_accuracy:.4f}, Mean Cross-Entropy: {mean_cross_entropy:.4f}')
        for i, (acc, ce) in enumerate(zip(per_feature_accuracy, per_feature_cross_entropy)):
            print(f'Feature {i} Accuracy: {acc:.4f}, Cross-Entropy: {ce:.4f}')

        return metrics

def collect_probe_dataset(env_name, vecenv=None, policy=None, args=None, num_samples=50, device='cpu', feature_source='representation'):
    """
    Collect a dataset for probe training by running the policy in the environment.
    
    Args:
        env_name: Name of the environment
        vecenv: Vectorized environment instance (will be created if None)
        policy: Trained policy to extract hidden states from (will be loaded if None)
        args: Configuration arguments (will be loaded if None) 
        num_samples: Number of samples to collect
        device: Device to run inference on
        feature_source: 'representation' to use learned embeddings (default),
                        'observation' to use raw observations as features
        
    Returns:
        Tuple of (embeddings, labels) tensors where labels are position indices
    """

    print("Collecting probe dataset...")
    
    args = args or load_config(env_name)
    
    args['vec'] = dict(backend='Serial', num_envs=1)
    args['env']['random_sampling'] = True
    
    vecenv = vecenv or load_env(env_name, args)
    policy = policy or load_policy(args, vecenv, env_name)
    
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
    
    samples_collected = 0

    with torch.no_grad():
        while samples_collected < num_samples:
            ob_tensor = torch.as_tensor(ob).to(device) # (num_agents, obs_dim)
            
            if feature_source == 'observation':
                features = ob_tensor
            else:
                features = policy.policy.encode_observations(ob_tensor) # (num_agents, hidden_dim)
            
            counts = driver.get_ground_truth_counts() # (num_agents, 3)
            
            ground_truth = torch.tensor(counts, dtype=torch.long) # (num_agents, 3)
            
            embeddings_list.append(features.cpu())
            labels_list.append(ground_truth)
            samples_collected += 1
            
            if samples_collected % 10 == 0:
                print(f"Collected {samples_collected}/{num_samples} probe samples")
            
            logits, value = policy.forward_eval(ob_tensor, state)
            action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
            action = action.cpu().numpy().reshape(vecenv.action_space.shape)
            
            if isinstance(logits, torch.distributions.Normal):
                action = np.clip(action, vecenv.action_space.low, vecenv.action_space.high)
            
            ob = vecenv.step(action)[0]
    
    embeddings = torch.cat(embeddings_list, dim=0) # (N * num_agents, hidden_dim)
    labels = torch.cat(labels_list, dim=0)  # (N * num_agents, 3)
    
    print(f"Probe dataset collection complete: {len(embeddings)} samples")
    
    return embeddings, labels


def train_probe(embeddings, labels, hidden_dim=rep_size, num_feature_types=3, obs_size=7,
                       num_epochs=50, learning_rate=0.001, train_split=0.8,
                       early_stopping=True, patience=10, min_delta=0.0,
                       save_path='experiments/probe/probe.pt'):
    """
    Train a linear probe on collected data.
    
    Args:
        embeddings: Tensor of shape (N, hidden_dim)
        labels: Tensor of shape (N, num_feature_types) with count labels for each animal type
        hidden_dim: Dimensionality of embeddings (inferred if None)
        num_feature_types: Number of feature types to predict
        obs_size: Size of observation grid (obs_size x obs_size)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (trained_probe, metrics_dict)
    """
    if hidden_dim is None:
        hidden_dim = embeddings.shape[1]
    
    probe = Probe(hidden_dim=hidden_dim, num_feature_types=num_feature_types, obs_size=obs_size)
    
    n_samples = len(embeddings)
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_embeddings = embeddings[train_indices]
    train_labels = labels[train_indices]
    val_embeddings = embeddings[val_indices]
    val_labels = labels[val_indices]
    
    print(f"Training probe on {n_train} samples, validating on {len(val_indices)} samples")
    
    probe.train(
        train_embeddings, train_labels,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        early_stopping=early_stopping,
        patience=patience,
        min_delta=min_delta,
        save_path=save_path,
    )
    
    if len(val_indices) > 0:
        metrics = probe.evaluate(val_embeddings, val_labels)
    else:
        metrics = {"overall_accuracy": 0.0, "mean_cross_entropy": 0.0}
    
    return probe, metrics


def evaluate_probe(probe, env_name, vecenv=None, policy=None, args=None, num_samples=50, device='cpu', feature_source='representation'):
    """
    Evaluate a trained probe on fresh data.
    
    Args:
        probe: Trained Probe instance
        env_name: Name of the environment
        vecenv: Vectorized environment instance (optional)
        policy: Policy for hidden state extraction (optional)
        args: Configuration arguments (optional)
        num_samples: Number of evaluation samples
        device: Device for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    embeddings, labels = collect_probe_dataset(env_name, vecenv, policy, args, num_samples, device, feature_source)
    
    return probe.evaluate(embeddings, labels)

def checkpoint_probe(env_name, policy, accuracy, checkpoint, feature_source='representation'):
    args = load_config(env_name)
    embeddings, labels = collect_probe_dataset(env_name, args=args, policy=policy, num_samples=10, feature_source=feature_source)
    probe, metrics = train_probe(embeddings, labels, hidden_dim=None, num_epochs=20)
    accuracy[checkpoint] = metrics['overall_accuracy']

def update_probe(env_name, policy, accuracy, checkpoint, existing_probe=None, hidden_dim=rep_size, obs_size=7, feature_source='representation'):
    """
    Update an existing probe with new data or create one if it doesn't exist.
    
    Args:
        env_name: Name of the environment
        policy: Current policy to extract features from
        accuracy: Dictionary to store accuracy metrics
        checkpoint: Checkpoint metric
        existing_probe: Existing probe to continue training (or None to create new)
        hidden_dim: Dimensionality of embeddings
        obs_size: Size of observation grid
        feature_source: 'representation' or 'observation'
        
    Returns:
        Trained probe
    """
    args = load_config(env_name)
    embeddings, labels = collect_probe_dataset(env_name, args=args, policy=policy, num_samples=10, feature_source=feature_source)

    train = 5
    
    if existing_probe is None:
        probe = Probe(hidden_dim=hidden_dim, num_feature_types=3, obs_size=obs_size)
        train = 10
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
        num_epochs=train,
        learning_rate=0.001,
        val_embeddings=val_embeddings,
        val_labels=val_labels,
        early_stopping=False,
        save_path=None,
    )
    
    if len(val_embeddings) > 0:
        metrics = probe.evaluate(val_embeddings, val_labels)
        accuracy[checkpoint] = metrics
    
    return probe

if __name__ == "__main__":
    args = load_config("puffer_repgrid")
    embeddings, labels = collect_probe_dataset("puffer_repgrid", args=args, num_samples=50)
    probe, metrics = train_probe(embeddings, labels)
    evaluate_probe(probe, "puffer_repgrid")