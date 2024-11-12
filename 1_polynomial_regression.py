import numpy as np # Is it version 2.1 the one you are running?
import matplotlib.pyplot as plt 
import torch # Is it version 2.4 the one you are running?
import torch.nn as nn
import torch.optim as optim


def plot_polynomial(coeffs, z_range, color='b'):
    # Plot the polynomial defined by the coefficients
    z = np.linspace(z_range[0], z_range[1], 500)
    # coeffs[::-1] is used to have the coefficients in the correct order required by np.polyval
    p = np.polyval(coeffs[::-1], z)
    plt.figure(figsize=(12, 8))
    plt.plot(z, p, color=color, label='Polynomial')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title('Polynomial Plot')
    plt.legend()
    plt.savefig('polynomial_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset(coeffs, z_range, sample_size, sigma, seed=42):
    np.random.seed(seed)
    # Generate random values of z
    z = np.random.uniform(z_range[0], z_range[1], sample_size)
    # Create X.T = [1, z, z^2, z^3, z^4]
    X = np.column_stack([z**i for i in range(5)])
    # Create y = p(z) + noise
    # coeffs[::-1] is used to have the coefficients in the correct order required by np.polyval
    y = np.polyval(coeffs[::-1], z) + np.random.normal(0, sigma, sample_size)
    # Convert to torch tensors
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)

def visualize_data(X, y, coeffs, z_range, title=""):
    plt.figure(figsize=(12, 8))
    z = np.linspace(z_range[0], z_range[1], 500)
    p = np.polyval(coeffs[::-1], z)
    plt.plot(z, p, label='True Polynomial')
    # Use the second column of X as the z values
    plt.scatter(X[:, 1].numpy(), y.numpy(), alpha=0.6, label='Data')
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title(title)
    plt.legend()
    # Use the title to save the plot with the correct name
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_loss_curves(train_losses, val_losses, total_steps):
    # total_steps is the total number of steps in the training loop
    plt.figure(figsize=(14, 10))
    plt.semilogy(range(total_steps), train_losses, label='Training Loss', alpha=0.8)
    plt.semilogy(range(total_steps), val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Steps')
    plt.ylabel('Loss (log scale)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# Plot the true polynomial and the estimated polynomial
def plot_comparison(coeffs_true, coeffs_est, z_range):
    z = np.linspace(z_range[0], z_range[1], 500)
    p_true = np.polyval(coeffs_true, z)
    p_est = np.polyval(coeffs_est, z)
    plt.figure(figsize=(12, 8))
    plt.plot(z, p_true, label='True Polynomial', linewidth=2)
    plt.plot(z, p_est, linestyle='--', label='Estimated Polynomial', linewidth=2)
    plt.xlabel('z')
    plt.ylabel('p(z)')
    plt.title('True vs Estimated Polynomial')
    plt.legend()
    plt.savefig('polynomial_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
# Plot the convergence of the parameters  
def plot_parameter_convergence(parameter_history, true_params):
    
    parameter_history = np.array(parameter_history)
    plt.figure(figsize=(14, 10))
    colors = ['b', 'g', 'r', 'c', 'black']  # For clarity of the plot
    for i in range(5):
        plt.plot(parameter_history[:, i], label=f'w[{i}]', color=colors[i])
        plt.axhline(y=true_params[i], linestyle='--', label=f'True w[{i}]', color=colors[i])
    plt.xlabel('Steps')
    plt.ylabel('Parameter Values')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.savefig('parameter_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    '''
    Code for Q2
    '''
    coeffs = np.array([1, -1, 5, -0.1, 1/30])
    z_range = [-4, 4]
    plot_polynomial(coeffs, z_range, color='b')
    
    '''
    Code for Q4
    '''
    # Coefficients in ascending order of degree as requested
    coeffs = np.array([1, -1, 5, -0.1, 1/30])
    z_range = [-2, 2]
    sigma = 0.5
    sample_size_train = 500
    sample_size_eval = 500

    X_train, y_train = create_dataset(coeffs, z_range, sample_size_train, sigma, seed=0)
    X_eval, y_eval = create_dataset(coeffs, z_range, sample_size_eval, sigma, seed=1)

    '''
    Code for Q5
    '''
    visualize_data(X_train, y_train, coeffs, z_range, "Training Data")
    visualize_data(X_eval, y_eval, coeffs, z_range, "Validation Data")
    
    '''
    Code for Q6
    '''
    model = nn.Linear(5, 1, bias = False)
    criterion = nn.MSELoss()
    # Best learning rate found
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    n_epochs = 5 
    steps_per_epoch = 300
    total_steps = n_epochs * steps_per_epoch

    train_losses = []
    val_losses = []
    # Save the history of the parameters to plot the convergence
    parameter_history = []

    step = 0
    for epoch in range(n_epochs):
        for _ in range(steps_per_epoch):
            # Training loop
            model.train()
            optimizer.zero_grad()
            # Compute the output of the model
            y_hat = model(X_train)
            # Compute the loss
            loss = criterion(y_hat, y_train)
            # Compute the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
            
            # Validation loop
            with torch.no_grad():
                model.eval()
                y_hat_val = model(X_eval)
                loss_val = criterion(y_hat_val, y_eval)
            
            # Save the losses and the parameters
            train_losses.append(loss.item())
            val_losses.append(loss_val.item())
            parameter_history.append(model.weight.data.numpy().flatten())
            
            # Print the losses at each epoch
            step += 1
            if step % steps_per_epoch == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Step [{step}/{total_steps}], '
                      f'Train Loss: {loss.item():.4f}, Val Loss: {loss_val.item():.4f}')
       
    '''
    Code for Q7
    '''   
    # Plot the loss curves of the training and validation sets 
    plot_loss_curves(train_losses, val_losses, total_steps)

    '''
    Code for Q8
    '''   
    # Plot the final result: true polynomial vs estimated polynomial
    estimated_coeffs = model.weight.data.numpy().flatten()
    plot_comparison(coeffs, estimated_coeffs, z_range)
    
    '''
    Code for Q9
    '''
    # Plot the convergence of the parameters
    plot_parameter_convergence(parameter_history, coeffs)

    print("True coefficients:", coeffs)
    print("Estimated coefficients:", estimated_coeffs)
    
    print("Training done, with an evaluation loss of {}".format(loss_val.item()))

    # Get the final value of the parameters
    print("Final w:", model.weight, "Final b:\n ", model.bias)
    
assert np.version.version=="2.1.0"