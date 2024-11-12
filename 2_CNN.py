

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

def q1_seed_experiment():
    '''
    Q1
    '''
    # Set initial seed for reproducibility
    torch.manual_seed(manual_seed)
    print("First run:")
    print(torch.randint(1, 10, (1,1)))
    
    #  Second run - random state advances
    print("Second run in same sequence:")
    print(torch.randint(1, 10, (1,1)))
    
    # Reset seed - we can see reproducibility
    torch.manual_seed(manual_seed)
    print("\nAfter resetting seed:")
    print(torch.randint(1, 10, (1,1)))

def load_and_inspect_data():
    '''
    Q2
    '''
    transform = transforms.ToTensor()
    
    # Load training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Display one image per class for training set
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx, class_name in enumerate(classes):
        class_indices = [i for i, (_, label) in enumerate(trainset) if label == idx]
        img, _ = trainset[class_indices[0]]
        img = img.numpy().transpose((1, 2, 0))
        axes[idx].imshow(img)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')
        
    fig.suptitle('CIFAR-10 Classes in Training Set')
    plt.tight_layout()
    plt.savefig('cifar10_classes_training.png')
    plt.show()

    # Display one image per class for test set
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx, class_name in enumerate(classes):
        class_indices = [i for i, (_, label) in enumerate(testset) if label == idx]
        img, _ = testset[class_indices[0]]
        img = img.numpy().transpose((1, 2, 0))
        axes[idx].imshow(img)
        axes[idx].set_title(class_name)
        axes[idx].axis('off')
        
    fig.suptitle('CIFAR-10 Classes in Test Set')
    plt.tight_layout()
    plt.savefig('cifar10_classes_test.png')
    plt.show()
    
    # Plot class distribution histograms
    train_labels = [label for _, label in trainset]
    plt.figure(figsize=(10, 5))
    plt.hist(train_labels, bins=len(classes), rwidth=0.8)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.title('Distribution of Classes in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.grid(True)
    plt.savefig('cifar10_class_distribution_train.png')
    plt.show()
    
    test_labels = [label for _, label in testset]
    plt.figure(figsize=(10, 5))
    plt.hist(test_labels, bins=len(classes), rwidth=0.8)
    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.title('Distribution of Classes in Test Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.grid(True)
    plt.savefig('cifar10_class_distribution_test.png')
    plt.show()
    
    # Add print statements for clearer analysis
    print("Dataset sizes:")
    print(f"Training set: {len(trainset)} images")
    print(f"Test set: {len(testset)} images")
    print("\nClass distribution in training set:")
    for i, c in enumerate(classes):
        count = train_labels.count(i)
        print(f"{c}: {count} images ({count/len(trainset)*100:.1f}%)")
        
    return trainset, testset

def analyze_dataset_properties():
    '''
    Q3
    '''
    # Get a sample image
    sample_img, _ = trainset_raw[0]
    
    print("Dataset Analysis:")
    print(f"(i) Type of each element: {type(sample_img)}")
    print(f"(ii) Current data type: {sample_img.dtype}")
    print(f"(iii) Image dimensions: {sample_img.shape}")
    print(f"(iv) Meaning of dimensions: {sample_img.shape[0]} channels, "
          f"{sample_img.shape[1]}x{sample_img.shape[2]} pixels")
    
    # Additional helpful information
    print("\nAdditional properties:")
    print(f"Value range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
    print(f"Mean value: {sample_img.mean():.3f}")
    print(f"Standard deviation: {sample_img.std():.3f}")

def normalize_dataset():
    '''
    Q4
    '''
    
    # Calculate mean and std for each channel
    means = []
    stds = []
    for i in range(3):  # 3 channels (RGB)
        channel_pixels = torch.stack([img[i, :, :] for img, _ in trainset_raw])
        means.append(float(channel_pixels.mean()))
        stds.append(float(channel_pixels.std()))
    
    print("Original dataset statistics:")
    print(f"Channel means: {means}")
    print(f"Channel stds: {stds}")
    
    # For dataset normalization, transforms.Normalize uses the formula:
    # normalized = (x - mean) / std
    transform_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=means,  # uses calculated means
            std=stds     # uses calculated stds
        )
    ])
    
    # Load datasets with normalization
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_normalize)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_normalize)
    
    # Verify normalization by computing stats over the entire training set
    all_channels = []
    for img, _ in trainset:
        all_channels.append(img)
    all_channels = torch.stack(all_channels)
    
    print("\nNormalized image statistics (full dataset):")
    for i, channel in enumerate(['Red', 'Green', 'Blue']):
        channel_mean = torch.mean(all_channels[:, i, :, :])
        channel_std = torch.std(all_channels[:, i, :, :])
        print(f"{channel} channel - Mean: {channel_mean:.4f}, Std: {channel_std:.4f}")
    
    return trainset, testset

def create_validation_set(testset, val_split=0.5):
    '''
    Q5
    '''
    # Add more generality by allowing for different split sizes
    total_size = len(testset)
    val_size = int(val_split * total_size)
    test_size = total_size - val_size
    
    # Use random_split with fixed seed for reproducibility
    validation_set, new_test_set = torch.utils.data.random_split(
        testset, 
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Created validation set with {len(validation_set)} samples")
    print(f"Remaining test set has {len(new_test_set)} samples")
    
    return validation_set, new_test_set

class ConvNet(nn.Module):
    '''
    Q6
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate final feature map size
        # Input: 32x32 -> conv1: 30x30 -> conv2: 28x28 -> pool1: 14x14
        # -> conv3: 12x12 -> conv4: 10x10 -> pool2: 5x5
        self.fc = nn.Linear(128 * 5 * 5, 10)
        
    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second block
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten and fully connected
        x = x.view(-1, 128 * 5 * 5)
        x = self.fc(x)
        return x

def test_network(model, test_loader, device):
    '''
    Helper function to test the network
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_model(model, train_loader, validation_loader, device, epochs=4, lr=0.03):
    '''
    Q7
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1)
    
    # For storing metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    n = 100  # Print every n steps
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate total loss properly
            total_train_loss += loss.item()
            num_batches += 1
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if (i + 1) % n == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'TRAIN Loss: {total_train_loss/num_batches:.4f}, '
                      f'TRAIN Acc: {100.*correct/total:.2f}%')
        
        # Calculate average training loss properly
        epoch_train_loss = total_train_loss / num_batches
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
                
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate average validation loss properly
        epoch_val_loss = val_loss / val_batches
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(val_acc)
        
        # Save metrics
        train_losses.append(epoch_train_loss)
        train_accs.append(train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Training Loss: {epoch_train_loss:.4f}')
        print(f'Training Acc: {train_acc:.2f}%')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Validation Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}\n')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    return train_losses, train_accs, val_losses, val_accs

def plot_training_results(train_losses, train_accs, val_losses, val_accs, filename):
    '''
    Q8
    '''
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy', marker='o')
    plt.plot(val_accs, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

class ImprovedConvNet(nn.Module):
    '''
    Q9
    '''
    def __init__(self, dropout_rate=0.2):
        super(ImprovedConvNet, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 10)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # First block
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third block
        x = F.gelu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Fully connected
        x = x.view(-1, 512 * 4 * 4)
        x = F.gelu(self.bn6(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

def train_with_different_seeds():
    '''
    Q10
    '''
    seeds = range(5, 10)
    test_accuracies = []
    
    for seed in seeds:
        print(f"\nTraining with seed {seed}")
        torch.manual_seed(seed)
        
        # Initialize model
        model = ConvNet().to(device)
        
        # Train the model
        train_model(model, train_loader, validation_loader, device)
        
        # Test accuracy
        test_acc = test_network(model, test_loader, device)
        test_accuracies.append(test_acc)
        print(f"Seed {seed} test accuracy: {test_acc:.2f}%")
    
    # Print summary
    print("\nSummary of test accuracies:")
    for seed, acc in zip(seeds, test_accuracies):
        print(f"Seed {seed}: {acc:.2f}%")
    print(f"Mean accuracy: {sum(test_accuracies)/len(test_accuracies):.2f}%")
    print(f"Std deviation: {np.std(test_accuracies):.2f}%")

if __name__ == "__main__":
    '''
    DON'T MODIFY THE SEED!
    '''
    # Set the seed for reproducibility
    manual_seed = 42
    torch.manual_seed(manual_seed)
    #ensure reproducibility and consistency across different runs.
    np.random.seed(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    '''
    Q1 - Code
    '''
    print("\nQuestion 1: Seed Experiment")
    print("-" * 50)
    q1_seed_experiment()

    '''
    Q2 - Code
    '''
    print("\nQuestion 2: Data Loading and Inspection")
    print("-" * 50)
    trainset_raw, testset_raw = load_and_inspect_data()

    '''
    Q3 - Code
    '''
    print("\nQuestion 3: Dataset Properties Analysis")
    print("-" * 50)
    analyze_dataset_properties()

    '''
    Q4 - Code
    '''
    print("\nQuestion 4: Dataset Normalization")
    print("-" * 50)
    trainset, testset = normalize_dataset()

    '''
    Q5 - Code
    '''
    print("\nQuestion 5: Creating Validation Set")
    print("-" * 50)
    # Create validation set by splitting test set
    validation_set, test_set = create_validation_set(testset)
    


    '''
    Q6 - Code
    '''
    print("\nQuestion 6: ConvNet Implementation")
    print("-" * 50)
    # Initialize model
    model = ConvNet()
    print("ConvNet Architecture:")
    print(model)

    # Create data loaders
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
    validation_loader = torch.utils.data.DataLoader(validation_set, 
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=2)


    # Test with a sample batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    sample_batch = next(iter(test_loader))[0].to(device)
    output = model(sample_batch)
    print(f"\nSample batch shape: {sample_batch.shape}")
    print(f"Output shape: {output.shape}")


    '''
    Q7 - Code: Training
    '''
    print("\nQuestion 7: Training the Network")
    print("-" * 50)
    
    # Train the model
    print(f"Training on {device}")
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, validation_loader, device)
    
    # Test accuracy
    test_accuracy = test_network(model, test_loader, device)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    

    '''
    Q8 - Code
    '''
    print("\nQuestion 8: Plotting Training Results")
    print("-" * 50)
    plot_training_results(train_losses, train_accs, val_losses, val_accs, 
                         'training_metrics_before.png')
    
    '''
    Q9 - Code
    '''
    print("\nQuestion 9: Training Improved Model")
    print("-" * 50)
    
    # Save models and results
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save base model
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    }, 'models/base_model.pth')
    
    # Reset states and seed before training the improved model
    torch.cuda.empty_cache()  # Clear GPU memory if using GPU
    torch.manual_seed(manual_seed)  # Reset seed
    np.random.seed(manual_seed)  # Reset numpy seed
    
    # Initialize improved model
    model = ImprovedConvNet().to(device)
    
    print("Improved ConvNet Architecture:")
    print(model)
    
    # Train improved model
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, validation_loader, device, epochs=4, lr=0.1)
    
    # Test improved model
    test_accuracy = test_network(model, test_loader, device)
    print(f"\nImproved Model Test Accuracy: {test_accuracy:.2f}%")
    
    # Plot training results
    plot_training_results(train_losses, train_accs, val_losses, val_accs, 
                         'training_metrics_improved.png')

   
    '''
    Q10 - Code
    '''
    print("\nQuestion 10: Training with Different Seeds")
    print("-" * 50)
    
    # Create arrays to store results
    seed_results = []
    seeds = range(5, 10)
    
    for seed in seeds:
        print(f"\nTraining with seed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize model
        model = ConvNet().to(device)
        
        # Train model
        _, _, _, _ = train_model(model, train_loader, validation_loader, device)
        
        # Test accuracy
        acc = test_network(model, test_loader, device)
        seed_results.append(acc)
        print(f"Seed {seed} test accuracy: {acc:.2f}%")
    
    # Print summary statistics
    print("\nSeed Analysis Summary:")
    print("-" * 20)
    print(f"Mean accuracy: {np.mean(seed_results):.2f}%")
    print(f"Std deviation: {np.std(seed_results):.2f}%")
    print(f"Min accuracy: {np.min(seed_results):.2f}%")
    print(f"Max accuracy: {np.max(seed_results):.2f}%")
    
    # Plot seed results
    plt.figure(figsize=(10, 5))
    plt.plot(seeds, seed_results, 'bo-')
    plt.title('Test Accuracy vs. Different Seeds')
    plt.xlabel('Seed Value')
    plt.ylabel('Test Accuracy (%)')
    plt.grid(True)
    plt.savefig('seed_analysis.png')
    plt.show()