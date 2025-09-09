import torch
import logging

class Probe():
    def __init__(self, hidden_dim: int = 128, num_feature_types: int = 3)  -> None:
        '''
        Initialize the probe.
        :param hidden_dim: The dimensionality of the hidden layer of the probe.
        :param num_feature_types: The number of feature types to predict counts for.
        :return: None
        '''

        self.probe = torch.nn.Linear(hidden_dim, num_feature_types) # predicts count for each feature type
        self.num_feature_types = num_feature_types


    def train(self, data_embeddings: torch.Tensor, labels: torch.Tensor, num_epochs: int = 10,
              learning_rate: float = 0.001, batch_size: int = 32) -> None:
        '''
        Train the probe on the embeddings of data from the model.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape (N, num_feature_types), where each row contains the count of each feature type.
        :param num_epochs: The number of epochs to train the probe for. An epoch is one pass through the entire dataset.
        :param learning_rate: How fast the probe learns.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: None
        '''

        criterion = torch.nn.MSELoss() # mse loss for count prediction
        optimizer = torch.optim.Adam(self.probe.parameters(), lr=learning_rate)

        logging.info('Training the probe...')
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, len(data_embeddings), batch_size):

                batch_embeddings = data_embeddings[i:i+batch_size].detach() # embeddings (N, D), detach to avoid gradients
                batch_labels = labels[i:i+batch_size].float() # labels (N, num_feature_types)

                outputs = self.probe(batch_embeddings) # (N, num_feature_types)

                loss = criterion(outputs, batch_labels) # mse loss

                optimizer.zero_grad()
                loss.backward() # update params
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

        logging.info('Training complete.')


    def predict(self, data_embeddings: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        '''
        Get the probe's predictions on the embeddings from the model, for unseen data.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A tensor of shape (N, num_feature_types), where each row contains predicted counts for each feature type.
        '''

        self.probe.eval()
        all_predicted = []

        with torch.no_grad():
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size]
                outputs = self.probe(batch_embeddings) # (batch_size, num_feature_types)
                all_predicted.append(outputs)

        return torch.cat(all_predicted, dim=0)


    def evaluate(self, data_embeddings: torch.Tensor, labels: torch.Tensor, batch_size: int = 32) -> dict:
        '''
        Evaluate the probe's performance by testing it on unseen data.
        :param data_embeddings: A tensor of shape (N, D), where N is the number of samples and D is the dimensionality of the embeddings.
        :param labels: A tensor of shape (N, num_feature_types), where each row contains the true counts for each feature type.
        :param batch_size: Used to batch the data for computational efficiency.
        :return: A dictionary containing evaluation metrics.
        '''

        self.probe.eval()
        all_predicted = []
        all_labels = []

        with torch.no_grad():
            for i in range(0, len(data_embeddings), batch_size):
                batch_embeddings = data_embeddings[i:i+batch_size]
                batch_labels = labels[i:i+batch_size].float()

                outputs = self.probe(batch_embeddings)
                
                all_predicted.append(outputs)
                all_labels.append(batch_labels)

        all_predicted = torch.cat(all_predicted, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        mse = torch.nn.functional.mse_loss(all_predicted, all_labels).item()
        mae = torch.nn.functional.l1_loss(all_predicted, all_labels).item() # linear error
        
        per_feature_mse = [] # mse per feature type
        for i in range(self.num_feature_types):
            feature_mse = torch.nn.functional.mse_loss(all_predicted[:, i], all_labels[:, i]).item()
            per_feature_mse.append(feature_mse)

        metrics = {
            'mse': mse,
            'mae': mae,
            'per_feature_mse': per_feature_mse,
            'rmse': mse ** 0.5
        }

        logging.info(f'Probe MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {metrics["rmse"]:.4f}')
        for i, feature_mse in enumerate(per_feature_mse):
            logging.info(f'Feature {i} MSE: {feature_mse:.4f}')

        return metrics

def collect_probe_dataset(env, policy, num_samples=10000, device='cpu'):
    """
    Collect a dataset for probe training by running the policy in the environment.
    
    Args:
        env: Environment instance (should have random_sampling=True)
        policy: Trained policy to extract hidden states from
        num_samples: Number of samples to collect
        device: Device to run inference on
        
    Returns:
        Tuple of (embeddings, labels) tensors
    """
    embeddings = []
    labels = []
    
    return embeddings, labels


def train_probe_on_data(embeddings, labels, hidden_dim=128, num_feature_types=3, 
                       num_epochs=100, learning_rate=0.001, train_split=0.8):
    """
    Train a linear probe on collected data.
    
    Args:
        embeddings: Tensor of shape (N, hidden_dim)
        labels: Tensor of shape (N, num_feature_types)
        hidden_dim: Dimensionality of embeddings (inferred if None)
        num_feature_types: Number of feature types to predict
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (trained_probe, metrics_dict)
    """
    if hidden_dim is None:
        hidden_dim = embeddings.shape[1]
    
    probe = Probe(hidden_dim=hidden_dim, num_feature_types=num_feature_types)
    
    n_samples = len(embeddings)
    n_train = int(n_samples * train_split)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_embeddings = embeddings[train_indices]
    train_labels = labels[train_indices]
    val_embeddings = embeddings[val_indices]
    val_labels = labels[val_indices]
    
    logging.info(f"Training probe on {n_train} samples, validating on {len(val_indices)} samples")
    
    probe.train(train_embeddings, train_labels, 
                num_epochs=num_epochs, learning_rate=learning_rate)
    
    if len(val_indices) > 0:
        metrics = probe.evaluate(val_embeddings, val_labels)
    else:
        metrics = {"mse": 0.0, "mae": 0.0}
    
    return probe, metrics


def evaluate_probe(probe, env, policy, num_samples=1000, device='cpu'):
    """
    Evaluate a trained probe on fresh data.
    
    Args:
        probe: Trained Probe instance
        env: Environment for data collection
        policy: Policy for hidden state extraction
        num_samples: Number of evaluation samples
        device: Device for inference
        
    Returns:
        Dictionary of evaluation metrics
    """
    embeddings, labels = collect_probe_dataset(env, policy, num_samples, device)
    
    return probe.evaluate(embeddings, labels)