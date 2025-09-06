"""
Dataset loading and preprocessing utilities for federated learning experiments.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import numpy as np
import heapq
from typing import Tuple, List, Optional


class TransformedSubset(Dataset):
    """Wrapper to apply transforms to a subset of a dataset."""
    
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)


class CIFAR100DataManager:
    """
    Manages CIFAR-100 dataset loading and preprocessing for federated learning.
    """
    
    def __init__(self, data_dir: str = "./data", batch_size: int = 128, 
                 num_workers: int = 4, download: bool = True):
        """
        Initialize the CIFAR-100 data manager.
        
        Args:
            data_dir: Directory to store/load the dataset
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            download: Whether to download the dataset if not present
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # Define transformations with ImageNet normalization
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                               (0.229, 0.224, 0.225))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), 
                               (0.229, 0.224, 0.225))
        ])
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load CIFAR-100 datasets without transformations (raw data)."""
        # Load raw datasets without transforms
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=self.download,
            transform=None
        )
        
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=self.download,
            transform=None
        )
    
    def get_centralized_loaders(self, val_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get centralized training, validation, and test data loaders.
        
        Args:
            val_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Check if we already have a validation subset from federated dataset creation
        if hasattr(self, '_val_subset'):
            # Use the pre-split validation subset
            val_subset = self._val_subset
            # Create a dummy train subset (not used for federated learning)
            train_subset = val_subset  # This won't be used
        else:
            # Split training data into train and validation
            val_size = int(len(self.train_dataset) * val_split)
            train_size = len(self.train_dataset) - val_size
            
            train_subset, val_subset = random_split(
                self.train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Apply transforms using TransformedSubset wrapper
        train_transformed = TransformedSubset(train_subset, self.train_transform)
        val_transformed = TransformedSubset(val_subset, self.test_transform)
        test_transformed = TransformedSubset(self.test_dataset, self.test_transform)
        
        train_loader = DataLoader(
            train_transformed, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_transformed, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_transformed, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_federated_datasets(self, num_clients: int, 
                                non_iid_degree: float = 0.0,
                                classes_per_client: Optional[int] = None,
                                val_split: float = 0.2) -> List[Subset]:
        """
        Create federated datasets for multiple clients.
        
        Args:
            num_clients: Number of clients to create datasets for
            non_iid_degree: Degree of non-IID distribution (0.0 = IID, 1.0 = highly non-IID)
            classes_per_client: Number of classes per client (overrides non_iid_degree if provided)
            val_split: Fraction of training data to reserve for validation (not distributed to clients)
            
        Returns:
            List of dataset subsets for each client
        """
        # First split the full training dataset into train and validation portions
        val_size = int(len(self.train_dataset) * val_split)
        train_size = len(self.train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            self.train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Store validation subset for later use
        self._val_subset = val_subset
        
        # Now distribute only the training portion among clients
        if classes_per_client is not None:
            # Use standard shard-based approach
            if classes_per_client == 100:  # IID case
                raw_datasets = create_iid_splits(train_subset, num_clients)
            else:
                raw_datasets = create_non_iid_splits(
                    train_subset, 
                    num_clients=num_clients, 
                    classes_per_client=classes_per_client
                )
            
            # Apply transforms to all client datasets using TransformedSubset
            transformed_datasets = []
            for raw_dataset in raw_datasets:
                transformed_dataset = TransformedSubset(raw_dataset, self.train_transform)
                transformed_datasets.append(transformed_dataset)
            
            return transformed_datasets
        elif non_iid_degree == 0.0:
            return self._create_iid_split_from_subset(train_subset, num_clients)
        else:
            return self._create_non_iid_split_from_subset(train_subset, num_clients, non_iid_degree)
    
    def _create_iid_split_from_subset(self, train_subset: Subset, num_clients: int) -> List[Subset]:
        """Create IID split of a subset among clients."""
        # Shuffle indices
        indices = list(range(len(train_subset)))
        np.random.shuffle(indices)
        
        # Split indices among clients
        client_indices = np.array_split(indices, num_clients)
        
        # Create subsets
        client_datasets = []
        for client_idx in range(num_clients):
            subset = Subset(train_subset, client_indices[client_idx])
            client_datasets.append(subset)
        
        return client_datasets
    
    def _create_non_iid_split_from_subset(self, train_subset: Subset, num_clients: int, non_iid_degree: float) -> List[Subset]:
        """Create non-IID split of a subset among clients."""
        # Group samples by class
        class_indices = {}
        for idx, (_, label) in enumerate(train_subset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Distribute classes among clients with controlled non-IID degree
        client_datasets = []
        for client_idx in range(num_clients):
            # Select subset of classes for this client
            num_classes = max(1, int(100 * (1 - non_iid_degree) / num_clients))
            selected_classes = np.random.choice(list(class_indices.keys()), 
                                             size=num_classes, replace=False)
            
            # Collect indices for selected classes
            client_indices = []
            for class_label in selected_classes:
                client_indices.extend(class_indices[class_label])
            
            # Create subset
            subset = Subset(train_subset, client_indices)
            client_datasets.append(subset)
        
        return client_datasets
    
    def _create_iid_split(self, num_clients: int) -> List[Subset]:
        """Create IID split of the dataset among clients."""
        # Shuffle indices
        indices = list(range(len(self.train_dataset)))
        np.random.shuffle(indices)
        
        # Split indices among clients
        client_indices = np.array_split(indices, num_clients)
        
        # Create subsets
        client_datasets = []
        for client_idx in range(num_clients):
            subset = Subset(self.train_dataset, client_indices[client_idx])
            client_datasets.append(subset)
        
        return client_datasets
    
    def _create_non_iid_split(self, num_clients: int, non_iid_degree: float) -> List[Subset]:
        """Create non-IID split of the dataset among clients."""
        # Group samples by class
        class_indices = {}
        for idx, (_, label) in enumerate(self.train_dataset):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Distribute classes among clients with controlled non-IID degree
        client_datasets = []
        for client_idx in range(num_clients):
            # Select subset of classes for this client
            num_classes = max(1, int(100 * (1 - non_iid_degree) / num_clients))
            selected_classes = np.random.choice(list(class_indices.keys()), 
                                             size=num_classes, replace=False)
            
            # Collect indices for selected classes
            client_indices = []
            for class_label in selected_classes:
                client_indices.extend(class_indices[class_label])
            
            # Create subset
            subset = Subset(self.train_dataset, client_indices)
            client_datasets.append(subset)
        
        return client_datasets
    
    def get_stratified_loader(self, dataloader: DataLoader, 
                            num_classes: int = 100, samples_per_class: int = 1) -> DataLoader:
        """
        Create a stratified data loader with balanced class representation.
        
        Args:
            dataloader: Source data loader
            num_classes: Number of classes to include
            samples_per_class: Number of samples per class
            
        Returns:
            Stratified data loader
        """
        # Collect samples by class
        class_samples = {}
        for batch_idx, (data, targets) in enumerate(dataloader):
            for sample_idx, target in enumerate(targets):
                target_item = target.item()
                if target_item not in class_samples:
                    class_samples[target_item] = []
                class_samples[target_item].append((data[sample_idx], target))
        
        # Select balanced samples
        selected_samples = []
        for class_label in range(min(num_classes, len(class_samples))):
            if class_label in class_samples:
                # Randomly select samples_per_class samples from this class
                class_data = class_samples[class_label]
                if len(class_data) >= samples_per_class:
                    selected_indices = np.random.choice(
                        len(class_data), size=samples_per_class, replace=False
                    )
                    for idx in selected_indices:
                        selected_samples.append(class_data[idx])
        
        # Create new dataset and loader
        if selected_samples:
            data_list, target_list = zip(*selected_samples)
            data_tensor = torch.stack(data_list)
            target_tensor = torch.stack(target_list)
            
            # Create a custom dataset
            class StratifiedDataset(torch.utils.data.Dataset):
                def __init__(self, data, targets):
                    self.data = data
                    self.targets = targets
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]
            
            stratified_dataset = StratifiedDataset(data_tensor, target_tensor)
            return DataLoader(stratified_dataset, batch_size=1, shuffle=False)
        else:
            # Fallback to original dataloader if no samples found
            return dataloader
    
    def get_adaptive_stratified_loader(self, dataloader: DataLoader, 
                                     classes_per_client: int) -> DataLoader:
        """
        Create stratified loader based on client's class distribution.
        
        Args:
            dataloader: Client's data loader
            classes_per_client: Number of classes this client sees (Nc)
        
        Returns:
            Stratified data loader with adaptive sampling:
            - IID (Nc=100): 100 classes × 1 sample = 100 examples
            - Non-IID (Nc<100): Nc classes × (100/Nc) samples = 100 examples
        """
        if classes_per_client == 100:  # IID setting
            return self.get_stratified_loader(dataloader, num_classes=100, samples_per_class=1)
        else:  # Non-IID setting
            samples_per_class = 100 // classes_per_client
            return self.get_stratified_loader(dataloader, num_classes=classes_per_client, samples_per_class=samples_per_class)


def create_iid_splits(dataset: Dataset, num_clients: int = 100, 
                     keep_transformations: bool = True, debug: bool = True) -> List[Subset]:
    """
    Create IID splits of the dataset among clients.
    IID is implemented as a special case of non-IID where classes_per_client = num_classes.
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        keep_transformations: Whether to keep dataset transformations
        debug: Whether to print debug information
        
    Returns:
        List of dataset subsets for each client
    """
    num_classes = 100  # CIFAR-100 has 100 classes
    return create_non_iid_splits(
        dataset=dataset, 
        num_clients=num_clients, 
        classes_per_client=num_classes, 
        debug=debug, 
        keep_transformations=keep_transformations
    )


def create_non_iid_splits(dataset: Dataset,  
                         num_clients: int = 100,
                         classes_per_client: int = 10,
                         random_state: int = 42,
                         keep_transformations: bool = True,
                         debug: bool = True) -> List[Subset]:
    """
    Create non-IID splits by assigning to each client shards of data
    from only classes_per_client classes.
    
    This implements the standard federated learning non-IID distribution approach
    used in FedAvg and related papers.
    
    Args:
        dataset: Dataset to split
        num_clients: Number of clients
        classes_per_client: Number of classes each client should see
        random_state: Random seed for reproducibility
        keep_transformations: Whether to keep dataset transformations
        debug: Whether to print debug information
        
    Returns:
        List of dataset subsets for each client
    """
    np.random.seed(random_state)
    
    # Get dataset information
    num_classes = 100  # CIFAR-100
    total_samples = len(dataset)
    
    # Validate parameters
    if classes_per_client > num_classes:
        raise ValueError(f"classes_per_client ({classes_per_client}) cannot be greater than num_classes ({num_classes})")
    
    if classes_per_client * num_clients < num_classes:
        raise ValueError(f"Not enough shards: {classes_per_client} * {num_clients} < {num_classes}")
    
    if debug:
        print(f"Creating non-IID splits: {num_clients} clients, {classes_per_client} classes per client")
        print(f"Total samples: {total_samples}, Classes: {num_classes}")
    
    # Get all targets to group by class
    targets = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        targets.append(target)
    targets = np.array(targets)
    
    # Group indices by class
    indices_by_label = {}
    for label in range(num_classes):
        indices_by_label[label] = np.where(targets == label)[0]
    
    # Calculate shards per class
    shards_per_class = (num_clients * classes_per_client) // num_classes
    if debug:
        print(f"Shards per class: {shards_per_class}")
    
    # Create shards for each class
    class_partitions = {}
    for label in range(num_classes):
        class_indices = indices_by_label[label]
        if len(class_indices) > 0:
            # Split class data into shards
            shard_size = len(class_indices) // shards_per_class
            remainder = len(class_indices) % shards_per_class
            
            shards = []
            start_idx = 0
            for i in range(shards_per_class):
                # Add one extra sample to first 'remainder' shards
                current_shard_size = shard_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_shard_size
                shards.append(class_indices[start_idx:end_idx])
                start_idx = end_idx
            
            class_partitions[label] = shards
        else:
            class_partitions[label] = []
    
    # Track availability of shards for each class
    availability = {label: len(shards) for label, shards in class_partitions.items()}
    
    # Assign shards to clients
    client_datasets = []
    for c_id in range(num_clients):
        # Pick classes with most remaining shards
        top_classes = heapq.nlargest(classes_per_client, availability.items(), key=lambda x: x[1])
        
        # Collect indices for this client
        client_indices = []
        client_classes = []
        
        for class_label, _ in top_classes:
            if availability[class_label] > 0:
                # Take one shard from this class
                shard = class_partitions[class_label].pop()
                client_indices.extend(shard)
                client_classes.append(class_label)
                availability[class_label] -= 1
        
        # Create subset for this client
        if client_indices:
            client_subset = Subset(dataset, client_indices)
            client_datasets.append(client_subset)
            
            if debug and c_id < 5:  # Print info for first 5 clients
                print(f"Client {c_id}: {len(client_indices)} samples, classes: {sorted(client_classes)}")
        else:
            # Fallback: give client a small random sample
            random_indices = np.random.choice(total_samples, size=min(50, total_samples), replace=False)
            client_subset = Subset(dataset, random_indices)
            client_datasets.append(client_subset)
            if debug:
                print(f"Client {c_id}: Fallback - {len(random_indices)} random samples")
    
    if debug:
        print(f"Created {len(client_datasets)} client datasets")
        # Print summary statistics
        total_client_samples = sum(len(dataset) for dataset in client_datasets)
        print(f"Total client samples: {total_client_samples}")
    
    return client_datasets
