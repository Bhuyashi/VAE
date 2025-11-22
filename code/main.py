import os

import torch
import torch.optim as optim

from config import Config
from dataloader import data_loader
from model import VariationalAutoEncoder, VAELoss
from visualization import plot_loss_curve, visualize_reconstruction, plot_latent_space, plot_latent_manifold, plot_latent_interpolation, plot_latent_interpolation_video, plot_multiple_interpolations_video

def train(trainloader, model, optimizer, loss_fn, config):
    model.train()
    losses = []  
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(trainloader):
            x = data.view(data.size(0), -1).to(config.DEVICE)

            # forward pass
            x_reconstructed, mu, log_var = model(x)
            _, recon, kl = loss_fn(x_reconstructed, x, mu, log_var)
            recon, kl = recon / float(data.size(0)), kl / float(data.size(0))

            # Warmup KL for intial epochs
            # WARMUP = 30
            # beta = 0.5 * (1 - torch.cos(torch.tensor(epoch / WARMUP * 3.14159)))
            # beta = min(beta.item(), 0.01)
            # recon_norm = recon / float(config.INPUT_DIM)
            # kl_norm = kl / float(config.Z_DIM)
            # loss = recon_norm + beta * kl_norm

            # reduce recon first then apply KL
            # if recon > 600:
            #     beta = 0.0
            # else:
            #     beta = 0.001
            # loss = recon + beta * kl

            loss = recon + kl

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # print loss
            if batch_idx % 500 == 0:
                print(f"Epoch {epoch+1}: Batch {batch_idx+1}/{len(trainloader)}, Loss: {loss.item()}, Recon: {recon.item()}, KL: {kl.item()}")

        avg_loss = train_loss / len(trainloader)
        losses.append(avg_loss)  
        print("-"*50)
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Average Loss: {avg_loss}")
        print("-"*50)

    # save model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(config.MODEL_SAVE_PATH, f"model_{config.DATASET}_{config.H_DIM}_{config.Z_DIM}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': config.INPUT_DIM,
        'hidden_dim': config.H_DIM,
        'z_dim': config.Z_DIM
    }, model_path)
    print(f"Model saved to {model_path}")

    return model, losses

def inference(testloader, model, config):
    print("\n" + "="*50)
    print("INFERENCE")
    print("="*50 + "\n")
    
    print("--- Reconstruction ---")
    visualize_reconstruction(model, testloader, config.DEVICE, config.DATASET)
    
    if config.Z_DIM == 2:
        print("\n--- Latent Space Scatter Plot ---")
        plot_latent_space(model, testloader, config.DEVICE, config.DATASET)
        
        print("\n--- Latent Space Manifold (Grid Sampling) ---")
        plot_latent_manifold(model, config.DEVICE, config.DATASET, n=20)
    
    print("\n--- Latent Space Interpolation ---")
    plot_latent_interpolation(model, testloader, config.DEVICE, config.DATASET, n_steps=10)

    print("\n--- Creating Interpolation Video ---")
    plot_latent_interpolation_video(
        model, testloader, config.DEVICE, config.DATASET, pair_n=0,
        n_steps=50, fps=10, save_path=f'results/interpolation_{config.DATASET}_{config.H_DIM}_{config.Z_DIM}.mp4'
    )
    
    print("\n--- Creating Multiple Interpolations Video ---")
    plot_multiple_interpolations_video(
        model, testloader, config.DEVICE, config.DATASET,
        n_pairs=5, n_steps=30, fps=10, save_path=f'results/multi_interpolation_{config.DATASET}_{config.H_DIM}_{config.Z_DIM}.mp4'
    )

def load_pretrained_model(config):

    model_path = os.path.join(config.MODEL_SAVE_PATH, f'model_{config.DATASET}_{config.H_DIM}_{config.Z_DIM}.pth')
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    model = VariationalAutoEncoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        z_dim=checkpoint['z_dim']
    ).to(config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def main():
    config = Config()

    trainloader, testloader = data_loader(config.DATASET, config.BATCH_SIZE)
    model = VariationalAutoEncoder(config.INPUT_DIM, config.H_DIM, config.Z_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
    loss_fn = VAELoss()

    if config.LOAD_PRETRAINED:
        model = load_pretrained_model(config)
    else:
        model, losses = train(trainloader, model, optimizer, loss_fn, config)
        plot_loss_curve(losses)
    
    if config.INFERENCE:
        if not config.LOAD_PRETRAINED:
            model = load_pretrained_model(config)
        inference(testloader, model, config)


if __name__ == "__main__":
    main()