# Standard Non-IID Federated Learning Implementation

This document describes the implementation of standard non-IID data distributions for federated learning experiments, following the approach used in FedAvg and related papers.

## Overview

The implementation provides shard-based non-IID data distribution that creates realistic federated learning scenarios where clients have different data distributions. This is crucial for studying how federated learning algorithms perform under data heterogeneity.

## Standard Non-IID Scenarios

The implementation supports 5 standard scenarios used in federated learning research:

1. **IID**: `classes_per_client = 100` (all classes per client)
2. **Non-IID(1)**: `classes_per_client = 1` (extreme non-IID - each client sees only 1 class)
3. **Non-IID(5)**: `classes_per_client = 5` (moderate non-IID)
4. **Non-IID(10)**: `classes_per_client = 10` (mild non-IID)
5. **Non-IID(50)**: `classes_per_client = 50` (slight non-IID)

## Implementation Details

### Core Functions

#### `create_iid_splits(dataset, num_clients=100, keep_transformations=True, debug=True)`
Creates IID splits by calling `create_non_iid_splits` with `classes_per_client = 100`.

#### `create_non_iid_splits(dataset, num_clients=100, classes_per_client=10, random_state=42, keep_transformations=True, debug=True)`
Implements the standard shard-based non-IID distribution algorithm:

1. **Group by class**: Groups all dataset samples by their class labels
2. **Create shards**: Splits each class's data into multiple shards
3. **Calculate shards per class**: `shards_per_class = (num_clients * classes_per_client) // num_classes`
4. **Assign shards to clients**: Each client gets `classes_per_client` shards, one from each selected class
5. **Balanced assignment**: Uses a heap-based approach to ensure balanced distribution

### Key Features

- **Deterministic**: Uses `random_state` for reproducible splits
- **Balanced**: Ensures all classes are represented across clients
- **Flexible**: Can handle any `classes_per_client` value
- **Validation**: Checks if splits are mathematically possible
- **Debug info**: Shows which classes each client sees

## Usage

### Basic Usage

```python
from src.data.dataset_loader import CIFAR100DataManager, create_non_iid_splits

# Create data manager
data_manager = CIFAR100DataManager(batch_size=64, download=True)

# Method 1: Using CIFAR100DataManager
client_datasets = data_manager.create_federated_datasets(
    num_clients=100, 
    classes_per_client=10  # Non-IID(10) scenario
)

# Method 2: Using direct functions
client_datasets = create_non_iid_splits(
    dataset=data_manager.train_dataset,
    num_clients=100,
    classes_per_client=10,
    random_state=42,
    debug=True
)
```

### Running Standard Experiments

#### Single Scenario
```bash
# Run Non-IID(5) scenario
python src/experiments/run_baseline_federated.py --classes_per_client 5 --num_rounds 50

# Run IID scenario
python src/experiments/run_baseline_federated.py --classes_per_client 100 --num_rounds 50
```

#### All Standard Scenarios
```bash
# Run all 5 standard scenarios
python src/experiments/run_standard_non_iid.py --num_rounds 50 --num_clients 100
```

### Custom Scenarios
```bash
# Test specific scenarios only
python src/experiments/run_standard_non_iid.py --scenarios iid non_iid_1 non_iid_10
```

## Algorithm Details

### Shard-Based Distribution

The algorithm follows the standard approach:

1. **Input**: Dataset with N samples, C classes, K clients, M classes per client
2. **Shards per class**: `S = (K * M) // C`
3. **For each class**:
   - Split class data into S shards
   - Handle remainder samples by adding to first shards
4. **For each client**:
   - Select M classes with most remaining shards
   - Take one shard from each selected class
   - Update availability counts

### Example: Non-IID(10) with 100 clients

- Classes per client: 10
- Shards per class: `(100 * 10) // 100 = 10`
- Each class is split into 10 shards
- Each client gets 10 shards (one from each of 10 different classes)

## Expected Performance Impact

- **IID**: All clients see all classes → easier convergence, better performance
- **Non-IID(1)**: Extreme case → each client specializes in 1 class → harder convergence
- **Non-IID(5/10/50)**: Gradual difficulty → tests robustness of FedAvg to data heterogeneity

## Files Modified/Added

### Core Implementation
- `src/data/dataset_loader.py`: Added shard-based non-IID functions
- `src/experiments/run_baseline_federated.py`: Added `--classes_per_client` parameter
- `src/experiments/run_standard_non_iid.py`: New script for standard scenarios

### Testing
- `test_non_iid_implementation.py`: Test script to verify implementation

## Validation

The implementation includes several validation checks:

1. **Parameter validation**: Ensures `classes_per_client <= num_classes`
2. **Shard feasibility**: Ensures `classes_per_client * num_clients >= num_classes`
3. **Debug output**: Shows class distribution for first few clients
4. **Fallback handling**: Provides random samples if shard assignment fails

## Integration with Existing Code

The new implementation is backward compatible:

- Existing code using `non_iid_degree` continues to work
- New `classes_per_client` parameter overrides `non_iid_degree` when provided
- All existing experiment scripts remain functional

## Example Output

```
Creating non-IID splits: 100 clients, 10 classes per client
Total samples: 50000, Classes: 100
Shards per class: 10
Client 0: 500 samples, classes: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
Client 1: 500 samples, classes: [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
...
Created 100 client datasets
Total client samples: 50000
```

This implementation now provides complete coverage of the standard federated learning non-IID scenarios used in research, enabling systematic study of data heterogeneity effects on federated learning algorithms.
