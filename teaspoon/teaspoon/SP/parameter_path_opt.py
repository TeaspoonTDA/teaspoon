import torch
from torchdiffeq import odeint
import gudhi
from torch.optim.lr_scheduler import LambdaLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PersistentEntropy(lifetimes, normalize=False):
    """
    Computes the persistence entropy of a given set of lifetimes from a persistence diagram.

    Args:
        lifetimes (torch.Tensor): 1D tensor of lifetimes.
        normalize (bool): Whether to normalize the entropy on a scale from 0 to 1 (default: False).

    Returns:
        torch.Tensor: The persistence entropy.
    """
    if len(lifetimes) == 0:
        return torch.tensor(0.0)
    
    if len(lifetimes) == 1:
        return torch.tensor(0.0)
    
    L = torch.sum(lifetimes)
    p = lifetimes / L
    E = -torch.sum(p * torch.log2(p))
    
    if normalize:
        Emax = torch.log2(torch.tensor(len(lifetimes), dtype=torch.float64))
        E = E / Emax
    
    return E


class PathOptimizer:
    """
    Class to optimize the parameters of a system using a path optimization approach. The class allows for the customization of the loss function using persistence functions and forbidden regions.
    """

    def __init__(self, loss_terms, forbidden_regions=None, a=10.0):
        """
        Initializes the parameter path optimizer with options for custimizing the loss function using persistence functions and forbidden regions.

        Args:
            loss_terms (dict): Dictionary where keys are loss term names ('maxPers', 'entropy', 'perstot', 'reg') and values are their signs (+1 or -1). Use -1 to maximize and +1 to minimize.
            forbidden_regions (list): List of lambda functions defining forbidden       regions. Each function takes in the parameter array and 1D persistence diagram. The function should be defined such that it returns a positive value when the region is violated and a negative value otherwise.
            a (float): Coefficient for the forbidden regions. Default is 10.0. Larger values make the penalty for violating the forbidden region stronger but may lead to numerical instability.
        """
        valid_terms = {'maxPers', 'entropy', 'perstot', 'reg'}
        assert set(loss_terms.keys()).issubset(valid_terms), f"Invalid loss terms. Valid terms are: {valid_terms}."
        assert all(sign in [-1, 1] for sign in loss_terms.values()), "Signs must only contain +1 or -1."
        
        self.loss_terms = loss_terms
        self.forbidden_regions = forbidden_regions
        self.a = a

    def compute_loss(self, pts, params):
        """
        Computes the loss based on the selected components.

        Args:
            pts (torch.Tensor): Input points.
            params (tuple): Parameters.

        Returns:
            torch.Tensor: The computed loss.
            torch.Tensor: The persistence diagram.
        """
        rips = gudhi.RipsComplex(points=pts)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        i = st.flag_persistence_generators()

        rho, sigma = params

        if len(i[1]) > 0:
            i1 = torch.tensor(i[1][0], device=device)
        else:
            i1 = torch.empty((0, 4), dtype=int, device=device)

        diag1 = torch.norm(pts[i1[:, (0, 2)]] - pts[i1[:, (1, 3)]], dim=-1)

        loss = 0.0

        if 'maxPers' in self.loss_terms:  
            maxPers1 = torch.max((diag1[:, 1] - diag1[:, 0]).to(torch.float64))
            loss += self.loss_terms['maxPers'] * (maxPers1 / (maxPers1.detach() + 1e-8))

        if 'entropy' in self.loss_terms:  
            ent1 = PersistentEntropy((diag1[:, 1] - diag1[:, 0]).to(torch.float64), normalize=True)
            loss += self.loss_terms['entropy'] * (ent1 / (ent1.detach() + 1e-8))

        if 'perstot' in self.loss_terms:  
            perstot1 = torch.sum((diag1[:, 1] - diag1[:, 0]).to(torch.float64))
            loss += self.loss_terms['perstot'] * (perstot1 / (perstot1.detach() + 1e-8))

        if self.forbidden_regions:
            reg = 0.0
            for region in self.forbidden_regions:
                reg += torch.exp(torch.tensor(self.a, dtype=torch.float64) * region(params, diag1))
            loss += reg

        return loss, diag1
    
    def optimize(self, system, x0, t, optimizer, scheduler=None, ss_points=500):
        """
        Function to take one optimization step for the system parameters. 
        
        Args:
            system (torch.nn.Module): The system to optimize.
            x0 (torch.Tensor): Initial conditions.
            t (torch.Tensor): Time points starting at 0. 
            optimizer (torch.optim.Optimizer): Optimizer for the parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler (optional).
            ss_points (int): Number of points to take for the steady state trajectory.
        Returns:
            y: (torch.Tensor): The trajectory of the system.
            pd1: (torch.Tensor): The persistence diagram.
            loss: (torch.Tensor): The computed loss.
            grads: (torch.Tensor): The gradients with respect to the parameters.
        """

        # Compute the trajectory
        sol_main = odeint(system, x0, t)[-ss_points:,:] 
        
        # Compute gradient of the trajectory with respect to parameters
        loss, pd1 = self.compute_loss(sol_main, system.params)
        
        loss.backward()


        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(system.params, max_norm=1.0)
        grads = system.params.grad


        pd1 = pd1.cpu().detach().numpy()
        y = sol_main.cpu().detach().numpy()

        # Perform optimization step
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

        return y, pd1, loss, grads
    

if __name__ == "__main__":
    import torch
    from torchdiffeq import odeint
    from torch.optim.lr_scheduler import LambdaLR
    from teaspoon.SP.parameter_path_opt import PathOptimizer
    from IPython.display import clear_output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_min = 80
    x_max = 300
    y_min = 4
    y_max = 50

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
    path = [[lorenz.params[0].item(), lorenz.params[1].item()]]
    losses = []


    ###########
    forbidden_regions = [
        lambda params, pd1: (params[0] - x_max),  
        lambda params, pd1: -(params[0] - x_min),  
        lambda params, pd1: (params[1] - y_max),  
        lambda params, pd1: -(params[1] - y_min),
        lambda params, pd1: -((1/400)*(params[0]-190)**2+(1/25)*(params[1]-10)**2 - 1.0),  # Example constraint: unit disk of radius 5 centered at (190, 27)
    ]
    p_opt = PathOptimizer({
        'maxPers': -1}, forbidden_regions=forbidden_regions)


    # Define a list of lambda functions taking in the parameter array and 1D persistence diagram

    ###########


    for epoch in range(5):
        
        y, pd1, loss, grads = p_opt.optimize(lorenz, x0, t_main, optimizer, scheduler)

        d_rho, d_sigma = grads[0], grads[1]

        clear_output(wait=True)
        print(f"d(Loss)/d(sigma): {d_sigma.item():.5}") 
        print(f"d(Loss)/d(rho): {d_rho.item():.5}")
        print(f"Loss: {loss.item():.8}")
        print(f"Rho: {lorenz.params[0].item():.8} -- Sigma: {lorenz.params[1].item():.8}")

        path.append([lorenz.params[0].item(), lorenz.params[1].item()])
        
        losses.append(loss.item())

