import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import torch

def plot_loss_curve(losses):
    """
    Plot the training loss curve.
    
    Args:
        losses: List of average losses per epoch
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print("Loss curve displayed.")

def visualize_reconstruction(model, test_loader, device, dataset_name='mnist', n=10):
    """Visualize original vs reconstructed samples"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:n].view(n, -1).to(device)
        recon, _, _ = model(data)
        
        if dataset_name in ['mnist', 'fashion_mnist']:
            data = data.cpu().view(n, 28, 28)
            recon = recon.cpu().view(n, 28, 28)
            
            fig, axes = plt.subplots(2, n, figsize=(n, 2))
            for i in range(n):
                axes[0, i].imshow(data[i], cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(recon[i], cmap='gray')
                axes[1, i].axis('off')
            axes[0, 0].set_ylabel('Original', fontsize=12)
            axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
            plt.tight_layout()
            plt.show()
        else:
            print("Original:", data.cpu().numpy()[:5])
            print("Reconstructed:", recon.cpu().numpy()[:5])


def plot_latent_space(model, test_loader, device, dataset_name='mnist'):
    """Visualize the 2D latent space (only works if latent_dim=2)"""
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            latents.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    if latents.shape[1] == 2:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(f'Latent Space Visualization - {dataset_name.upper()}')
        plt.show()
    else:
        print(f"Latent space has {latents.shape[1]} dimensions. Cannot visualize (need 2D).")

def plot_latent_manifold(model, device, dataset_name='mnist', n=20, digit_size=28):
    """
    Visualize a 2D manifold of the latent space by sampling a grid
    and decoding to see how digits transition across the latent space.
    Only works for latent_dim=2.
    """
    model.eval()
    
    if dataset_name not in ['mnist', 'fashion_mnist']:
        print("Latent manifold visualization only supported for MNIST and Fashion-MNIST")
        return
    
    # Create a grid in latent space
    # Find reasonable bounds (typically [-3, 3] for standard normal)
    grid_range = 3
    grid_x = np.linspace(-grid_range, grid_range, n)
    grid_y = np.linspace(-grid_range, grid_range, n)[::-1]  # Reverse y so top is positive
    
    # Create figure
    figure = np.zeros((digit_size * n, digit_size * n))
    
    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                # Sample point in latent space
                z_sample = torch.FloatTensor([[xi, yi]]).to(device)
                
                # Decode
                x_decoded = model.decode(z_sample)
                digit = x_decoded.cpu().reshape(digit_size, digit_size).numpy()
                
                # Place in figure
                figure[i * digit_size: (i + 1) * digit_size,
                       j * digit_size: (j + 1) * digit_size] = digit
    
    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title(f'Latent Space Manifold - {dataset_name.upper()}\n'
              f'Each position shows decoded sample from that latent coordinate', 
              fontsize=14, pad=20)
    
    # Add axis labels to show latent coordinates
    plt.text(figure.shape[1] / 2, -20, f'Latent Dim 1: [{-grid_range:.1f}, {grid_range:.1f}]', 
             ha='center', fontsize=10)
    plt.text(-20, figure.shape[0] / 2, f'Latent Dim 2: [{-grid_range:.1f}, {grid_range:.1f}]', 
             va='center', rotation=90, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_latent_interpolation(model, test_loader, device, dataset_name='mnist', n_steps=10):
    """
    Interpolate between two random samples in latent space to show smooth transitions.
    Works for any latent dimension.
    """
    model.eval()
    
    if dataset_name not in ['mnist', 'fashion_mnist']:
        print("Interpolation visualization only supported for MNIST and Fashion-MNIST")
        return
    
    
    with torch.no_grad():
        # Get two random samples
        data, labels = next(iter(test_loader))
        data = data[:2].view(2, -1).to(device)
        
        # Encode to latent space
        mu, _ = model.encode(data)
        z_start, z_end = mu[0], mu[1]
        
        # Create interpolation
        alphas = np.linspace(0, 1, n_steps)
        
        fig, axes = plt.subplots(2, n_steps, figsize=(n_steps * 1.5, 3))
        
        for idx, alpha in enumerate(alphas):
            # Interpolate in latent space
            z_interp = (1 - alpha) * z_start + alpha * z_end
            
            # Decode
            x_decoded = model.decode(z_interp.unsqueeze(0))
            digit = x_decoded.cpu().reshape(28, 28).numpy()
            
            # Show interpolated image
            axes[0, idx].imshow(digit, cmap='gray')
            axes[0, idx].axis('off')
            axes[0, idx].set_title(f'{alpha:.1f}', fontsize=8)
        
        # Show original images in second row
        for idx in range(2):
            orig = data[idx].cpu().reshape(28, 28).numpy()
            if idx == 0:
                axes[1, 0].imshow(orig, cmap='gray')
                axes[1, 0].axis('off')
                axes[1, 0].set_title('Start', fontsize=10, fontweight='bold')
            else:
                axes[1, -1].imshow(orig, cmap='gray')
                axes[1, -1].axis('off')
                axes[1, -1].set_title('End', fontsize=10, fontweight='bold')
        
        # Hide middle axes in second row
        for idx in range(1, n_steps - 1):
            axes[1, idx].axis('off')
        
        plt.suptitle(f'Latent Space Interpolation - {dataset_name.upper()}', fontsize=14)
        plt.tight_layout()
        plt.show()

def plot_latent_interpolation_video(model, test_loader, device, dataset_name='mnist', pair_n=1,
                                     n_steps=50, fps=10, save_path='interpolation.mp4'):
    """
    Create an animated video interpolating between two random samples in latent space.
    Shows only the morphing interpolation without start/end images.
    Works for any latent dimension.
    
    Args:
        n_steps: Number of interpolation frames (more = smoother)
        fps: Frames per second for the video
        save_path: Where to save the video file
    """
    model.eval()
    
    if dataset_name not in ['mnist', 'fashion_mnist']:
        print("Interpolation visualization only supported for MNIST and Fashion-MNIST")
        return
    
    with torch.no_grad():
        # Get two random samples
        data, labels = next(iter(test_loader))
        data = data[(pair_n*2):(pair_n*2+2)].view(2, -1).to(device)
        
        # Encode to latent space
        mu, _ = model.encode(data)
        z_start, z_end = mu[0], mu[1]
        
        # Create interpolation points
        alphas = np.linspace(0, 1, n_steps)
        
        # Generate all frames
        frames = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z_start + alpha * z_end
            x_decoded = model.decode(z_interp.unsqueeze(0))
            digit = x_decoded.cpu().reshape(28, 28).numpy()
            frames.append(digit)
        
        # Create figure with single plot
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Setup plot
        im = ax.imshow(frames[0], cmap='gray', vmin=0, vmax=1)
        title = ax.set_title(f'Latent Space Interpolation\nα = 0.00', fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Animation function
        def update(frame_idx):
            im.set_array(frames[frame_idx])
            alpha = alphas[frame_idx]
            title.set_text(f'Latent Space Interpolation\nα = {alpha:.2f}')
            return [im, title]
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_steps, interval=1000/fps, blit=True)
        
        # Try to save as GIF
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=fps)
        
        plt.close()
        
        # Display the video if possible
        try:
            from IPython.display import Video, display
            display(Video(save_path, embed=True))
        except:
            print(f"GIF saved. Play it with a video player: {gif_path}")


def plot_multiple_interpolations_video(model, test_loader, device, dataset_name='mnist',
                                        n_pairs=5, n_steps=30, fps=10, save_path='multi_interpolation.mp4'):
    """
    Create a video showing multiple interpolations simultaneously in a grid.
    
    Args:
        n_pairs: Number of interpolation pairs to show
        n_steps: Number of interpolation frames per pair
        fps: Frames per second
        save_path: Where to save the video
    """
    model.eval()
    
    if dataset_name not in ['mnist', 'fashion_mnist']:
        print("Interpolation visualization only supported for MNIST and Fashion-MNIST")
        return
    
    with torch.no_grad():
        # Get multiple pairs
        data, labels = next(iter(test_loader))
        data = data[:n_pairs*2].view(n_pairs*2, -1).to(device)
        
        # Encode all samples
        mu, _ = model.encode(data)
        
        # Create pairs
        pairs = [(mu[i*2], mu[i*2+1]) for i in range(n_pairs)]
        
        # Generate all frames for all pairs
        alphas = np.linspace(0, 1, n_steps)
        all_frames = []
        
        for z_start, z_end in pairs:
            frames = []
            for alpha in alphas:
                z_interp = (1 - alpha) * z_start + alpha * z_end
                x_decoded = model.decode(z_interp.unsqueeze(0))
                digit = x_decoded.cpu().reshape(28, 28).numpy()
                frames.append(digit)
            all_frames.append(frames)
        
        # Create figure with grid
        fig, axes = plt.subplots(1, n_pairs, figsize=(n_pairs * 3, 3))
        if n_pairs == 1:
            axes = [axes]
        
        # Setup subplots
        ims = []
        titles = []
        for i, ax in enumerate(axes):
            im = ax.imshow(all_frames[i][0], cmap='gray', vmin=0, vmax=1)
            title = ax.set_title(f'Pair {i+1}: α = 0.00', fontsize=10)
            ax.axis('off')
            ims.append(im)
            titles.append(title)
        
        plt.suptitle(f'Multiple Latent Space Interpolations - {dataset_name.upper()}', fontsize=14)
        plt.tight_layout()
        
        # Animation function
        def update(frame_idx):
            alpha = alphas[frame_idx]
            for i in range(n_pairs):
                ims[i].set_array(all_frames[i][frame_idx])
                titles[i].set_text(f'Pair {i+1}: α = {alpha:.2f}')
            return ims + titles
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=n_steps, interval=1000/fps, blit=True)
        
        # Try to save
        gif_path = save_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"Animation saved. Play it with a video player: {gif_path}")
        plt.close()