

Parameter Path Optimization
=========================================================

This page contains the necessary functions for performing dynamical system parameter space optimization and allows for obtaining a path in the parameter space to minimize a defined loss function in terms of functions of persistence. 


.. automodule:: teaspoon.SP.parameter_path_opt
    :members: 

**Example**::

    import torch
    from torchdiffeq import odeint
    from torch.optim.lr_scheduler import LambdaLR
    from teaspoon.SP.parameter_path_opt import PathOptimizer
    from IPython.display import clear_output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set bounds for the parameters
    x_min = 80
    x_max = 300
    y_min = 4
    y_max = 50

    # Define the system of ODEs using PyTorch
    class LorenzSystem(torch.nn.Module):
        def __init__(self, params):
            super().__init__()
            self.params = torch.nn.Parameter(params)

        def forward(self, t, state):
            x, y, z = state
            rho, sigma = self.params
            dxdt = sigma * (y - x)
            dydt = x * (rho - z) - y
            dzdt = x * y - (8.0/3.0) * z
            return torch.stack([dxdt, dydt, dzdt])

    # Time settings
    t_main = torch.linspace(0, 10, 1000, device=device, dtype=torch.float64)   # Main simulation

    # Initial conditions
    x0 = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, device=device, dtype=torch.float64)

    # Combine parameters into a single vector
    params = torch.nn.Parameter(torch.tensor([190.0, 20.0], device=device))

    # Instantiate the system with the combined parameter vector
    lorenz = LorenzSystem(params).to(device)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.Adam([lorenz.params], lr=1.0) 
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.995**epoch)

    # Initialize lists for saving the path and losses
    path = [[lorenz.params[0].item(), lorenz.params[1].item()]]
    losses = []


    # Define the forbidden regions for the parameter path
    forbidden_regions = [
        lambda params, pd1: (params[0] - x_max),  
        lambda params, pd1: -(params[0] - x_min),  
        lambda params, pd1: (params[1] - y_max),  
        lambda params, pd1: -(params[1] - y_min),
        lambda params, pd1: -((1/400)*(params[0]-190)**2+(1/25)*(params[1]-10)**2 - 1.0),  # Example constraint: unit disk of radius 5 centered at (190, 27)
    ]

    # Initialize the PathOptimizer with the Lorenz system and forbidden regions
    p_opt = PathOptimizer({
        'maxPers': -1}, forbidden_regions=forbidden_regions)


    for epoch in range(5):
        # Perform the optimization step
        y, pd1, loss, grads = p_opt.optimize(lorenz, x0, t_main, optimizer, scheduler)

        # Extract the gradients with resspect to the parameters
        d_rho, d_sigma = grads[0], grads[1]

        # Print result of optimization step
        clear_output(wait=True)
        print(f"d(Loss)/d(sigma): {d_sigma.item():.5}") 
        print(f"d(Loss)/d(rho): {d_rho.item():.5}")
        print(f"Loss: {loss.item():.8}")
        print(f"Rho: {lorenz.params[0].item():.8} -- Sigma: {lorenz.params[1].item():.8}")

        # Save the path and the loss
        path.append([lorenz.params[0].item(), lorenz.params[1].item()])
        losses.append(loss.item())

Output of example (first step of optimization)

::

    d(Loss)/d(sigma): 0.97641
    d(Loss)/d(rho): 0.21593
    Loss: -1.0
    Rho: 189.0 -- Sigma: 19.0

.. note:: 
    Results may vary depending on the operating system for chaotic dynamics. The result shown is for demonstration purposes only.