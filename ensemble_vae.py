# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.ModuleList):
    def __init__(self, decoder_nets):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_nets = nn.ModuleList(decoder_nets)
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = torch.zeros_like(self.decoder_nets[0](z))

        for decoder_net in self.decoder_nets:
            means += decoder_net(z)

        means = means / len(self.decoder_nets)

        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break

def optimize_geodesic(model, z_start, z_intermediate, z_end, optimizer, num_steps=1000):

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                optimizer.zero_grad()
                
                # Rebuild the curve so the graph updates
                z_full_curve = torch.cat([z_start, z_intermediate, z_end], dim=0)
                
                # Compute the energy
                loss = compute_ensemble_energy(model, z_full_curve)
                
                # Backpropagate and take an optimizer step
                loss.backward()
                optimizer.step()

                # Report progress
                if step % 10 == 0:
                    pbar.set_description(f"step={step}, energy={loss.item():.2f}")

            except KeyboardInterrupt:
                print(f"Stopping optimization at step {step}")
                break
                
    # Return the final optimized curve (detached from gradient graph)
    final_curve = torch.cat([z_start, z_intermediate, z_end], dim=0).detach()
    return final_curve

# Computes the energy of a curve in latent space using the ensemble decoder nets
def compute_ensemble_energy(model, z_curve):

    energy = 0.0
    num_decoders = len(model.decoder.decoder_nets)

    for i in range(len(z_curve) - 1):

        indices = torch.randint(0, num_decoders, (2,))
        l = indices[0].item()
        k = indices[1].item()

        decoder_net_l = model.decoder.decoder_nets[l]
        decoder_net_k = model.decoder.decoder_nets[k]

        decoded_pts_l = decoder_net_l(z_curve)
        decoded_pts_k = decoder_net_k(z_curve)
        squared_diff = (decoded_pts_l[i] - decoded_pts_k[i+1])**2
        energy += squared_diff.sum()

    return energy

# Computes the energy of a curve in latent space using the ensemble decoder nets
# def compute_ensemble_energy(model, z_curve):
#     num_decoders = len(model.decoder.decoder_nets)
    
#     # 1. PRE-COMPUTE: Pass the curve through all decoders just ONCE.
#     # This creates a tensor of shape [num_decoders, num_t, 1, 28, 28]
#     all_decoded = torch.stack([net(z_curve) for net in model.decoder.decoder_nets])

#     energy = 0.0
    
#     # 2. Loop over each segment of the curve
#     for i in range(len(z_curve) - 1):
#         segment_energy = 0.0
        
#         # 3. EXACT EXPECTATION: Calculate the distance for ALL possible pairs of l and k
#         for l in range(num_decoders):
#             for k in range(num_decoders):
#                 squared_diff = (all_decoded[l, i] - all_decoded[k, i+1])**2
#                 segment_energy += squared_diff.sum()
        
#         # 4. Average the energy of the pairs and add to total energy
#         energy += segment_energy / (num_decoders * num_decoders)

#     return energy

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "eval_cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=30,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        # New way (Part B)
        # Create a list of 'N' separate decoders
        ensemble_list = [new_decoder() for _ in range(args.num_decoders)]

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(ensemble_list),
            GaussianEncoder(new_encoder()),
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":

        # New way (Part B)
        # Create a list of 'N' separate decoders
        ensemble_list = [new_decoder() for _ in range(args.num_decoders)]

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(ensemble_list),
            GaussianEncoder(new_encoder()),
        ).to(device)
        
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        # Create the ensemble list so the architecture matches the saved weights!
        ensemble_list = [new_decoder() for _ in range(args.num_decoders)]
        
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(ensemble_list),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        print("Extracting background latent space...")
        # We process the test loader to get z-coordinates and labels for the background of the plot
        all_z = []
        all_labels = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                z = model.encoder(x).mean
                all_z.append(z.cpu())
                all_labels.append(y.cpu())
        
        # Concatenate into big numpy arrays for matplotlib
        all_z = torch.cat(all_z, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Create the plot and draw the scatter background
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap='viridis', alpha=0.5, s=15)
        plt.colorbar(scatter, label="Digit Class")

        print(f"Optimizing {args.num_curves} geodesics...")

        # Get a batch of data to sample our endpoints from
        test_data_full = torch.cat([x for x, _ in mnist_test_loader], dim=0).to(device)

        torch.manual_seed(42)

        # Loop for user-specified number of curves
        for i in range(args.num_curves):
            print(f"\nOptimizing curve {i+1}/{args.num_curves}")
            
            # Pick random images for start and end
            idx = torch.randint(0, len(test_data_full), (2,))
            x_start = test_data_full[idx[0]:idx[0]+1]
            x_end = test_data_full[idx[1]:idx[1]+1]

            # Encode to get fixed endpoints
            z_start = model.encoder(x_start).mean.detach()
            z_end = model.encoder(x_end).mean.detach()

            # Initialize the straight line
            t = torch.linspace(0, 1, args.num_t).view(-1, 1).to(device)
            z_curve_init = (1 - t) * z_start + t * z_end
            z_intermediate = z_curve_init[1:-1].clone().detach().requires_grad_(True)

            # Setup optimizer for this specific curve
            curve_optimizer = torch.optim.Adam([z_intermediate], lr=1e-5)

            # Runs the optimization of the curve
            optimized_curve = optimize_geodesic(
                model=model,
                z_start=z_start,
                z_intermediate=z_intermediate,
                z_end=z_end,
                optimizer=curve_optimizer,
                num_steps=500
            )

            # Add curves to the plot
            # Convert tensors to numpy arrays for plotting
            init_np = z_curve_init.cpu().detach().numpy()
            opt_np = optimized_curve.cpu().detach().numpy()

            # Plot straight line in gray, dashed
            plt.plot(init_np[:, 0], init_np[:, 1], color='gray', linestyle='--', alpha=0.5)
            # Plot the optimized geodesic in red, solid
            plt.plot(opt_np[:, 0], opt_np[:, 1], color='red', linewidth=2, alpha=0.8)

        # Save the final image
        plt.title("Latent Space and Pull-back Geodesics")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.tight_layout()
        plt.savefig(f"{args.experiment_folder}/geodesics.png", dpi=300)
        plt.close()
        print(f"Done! Plot saved to {args.experiment_folder}/geodesics.png")

    elif args.mode == "eval_cov":
        print("Starting CoV Evaluation...")
        
        # FIX THE TEST POINTS
        torch.manual_seed(42) 
        test_data_full = torch.cat([x for x, _ in mnist_test_loader], dim=0).to(device)
        
        num_pairs = 1
        # num_pairs = 10
        fixed_pairs = []
        for _ in range(num_pairs):
            idx = torch.randint(0, len(test_data_full), (2,))
            fixed_pairs.append((test_data_full[idx[0]:idx[0]+1], test_data_full[idx[1]:idx[1]+1]))

        # We will test 1, 2 and 3 decoders
        decoders_list = [2]
        # decoders_list = [1, 2, 3]
        runs_list = range(1, args.num_reruns + 1)
        
        cov_euclidean = []
        cov_geodesic = []

        # Open a text file to save the report data
        report_file = open(f"{args.experiment_folder}/cov_report.txt", "w")
        report_file.write("--- Coefficient of Variation (CoV) Report ---\n\n")

        # LOOP OVER ARCHITECTURES
        for num_decs in decoders_list:
            print(f"\n==============================================")
            print(f"Evaluating Models with {num_decs} Decoder(s)")
            print(f"==============================================")
            
            euc_dists = torch.zeros((num_pairs, args.num_reruns))
            geo_dists = torch.zeros((num_pairs, args.num_reruns))
            
            # LOOP OVER THE 10 RETRAININGS
            for run_idx, run in enumerate(runs_list):
                model_path = f"{args.experiment_folder}/model_dec{num_decs}_run{run}/model.pt"
                
                ensemble_list = [new_decoder() for _ in range(num_decs)]
                model = VAE(GaussianPrior(M), GaussianDecoder(ensemble_list), GaussianEncoder(new_encoder())).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                
                # LOOP OVER THE 10 FIXED IMAGE PAIRS
                for pair_idx, (x_start, x_end) in enumerate(fixed_pairs):
                    z_start = model.encoder(x_start).mean.detach()
                    z_end = model.encoder(x_end).mean.detach()
                    
                    # Compute Euclidean Distance
                    euc_dist = torch.norm(z_start - z_end).item()
                    euc_dists[pair_idx, run_idx] = euc_dist
                    
                    # Compute Geodesic Distance
                    t = torch.linspace(0, 1, args.num_t).view(-1, 1).to(device)
                    z_curve_init = (1 - t) * z_start + t * z_end
                    z_intermediate = z_curve_init[1:-1].clone().detach().requires_grad_(True)
                    curve_optimizer = torch.optim.Adam([z_intermediate], lr=1e-4) 
                    
                    # Find the shortest path
                    optimized_curve = optimize_geodesic(
                        model=model, z_start=z_start, z_intermediate=z_intermediate, 
                        z_end=z_end, optimizer=curve_optimizer, num_steps=500 
                    )
                    
                    # Calculate the true Geodesic Distance using the ensemble pull-back metric
                    with torch.no_grad():
                        all_decoded = torch.stack([net(optimized_curve) for net in model.decoder.decoder_nets])

                    geo_dist = 0.0
                    
                    # Look at one segment of the curve at a time
                    for step in range(len(optimized_curve) - 1):
                        segment_energy = torch.tensor(0.0, device=device)
                        
                        # Exact expectation of the squared distance for the segment
                        for l in range(num_decs):
                            for k in range(num_decs):
                                squared_diff = (all_decoded[l, step] - all_decoded[k, step+1])**2
                                segment_energy += squared_diff.sum()
                        
                        segment_energy = segment_energy / (num_decs * num_decs)
                        
                        # Convert segment energy to segment length, then add to total
                        geo_dist += math.sqrt(segment_energy.item())
                    geo_dists[pair_idx, run_idx] = geo_dist
                    
            # CALCULATE CoV (Standard Deviation / Mean)
            euc_std = euc_dists.std(dim=1)
            euc_mean = euc_dists.mean(dim=1)
            euc_cov = (euc_std / euc_mean).mean().item() 
            
            geo_std = geo_dists.std(dim=1)
            geo_mean = geo_dists.mean(dim=1)
            geo_cov = (geo_std / geo_mean).mean().item() 
            
            cov_euclidean.append(euc_cov)
            cov_geodesic.append(geo_cov)
            
            # Print to console
            print(f"Results for {num_decs} Decoder(s):")
            print(f"  -> Average Euclidean CoV: {euc_cov:.5f}")
            print(f"  -> Average Geodesic CoV:  {geo_cov:.5f}\n")
            
            # Write to the text file
            report_file.write(f"Results for {num_decs} Decoder(s):\n")
            report_file.write(f"  Average Euclidean CoV: {euc_cov:.5f}\n")
            report_file.write(f"  Average Geodesic CoV:  {geo_cov:.5f}\n\n")

        # Close text file
        report_file.close()

        # Generate final plot
        plt.figure(figsize=(8, 6))
        plt.plot(decoders_list, cov_euclidean, marker='o', label='Euclidean Distance', linewidth=2)
        plt.plot(decoders_list, cov_geodesic, marker='s', label='Geodesic Distance', linewidth=2)
        plt.xticks(decoders_list)
        plt.xlabel("Number of Decoders in Ensemble")
        plt.ylabel("Average Coefficient of Variation (CoV)")
        plt.title("Distance Reliability vs. Ensemble Size")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{args.experiment_folder}/cov_analysis.png", dpi=300)
        plt.close()
        
        print(f"\nSuccess! ")
        print(f"- Plot saved to: {args.experiment_folder}/cov_analysis1.png")
        print(f"- Data saved to: {args.experiment_folder}/cov_report1.txt")