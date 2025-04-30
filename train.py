import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from models.dataloader import get_dataloader
from pandas import read_csv
import numpy as np
from models.combined_model import MultimodalModel
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

torch.manual_seed(42) # Sets the seed to 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def rmse(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()

def run_training(resume_from_checkpt=False, name = 'run', use_img = True, use_text = True, use_metadata = True):
    user_ratings = read_csv('raw_data/ratings.csv')
    metadata = read_csv('clean_data/metadata_IMDB.csv')
    dataset, dataloader = get_dataloader(plots_path = 'clean_data/plots_Imdb_corrected.csv', links_path = 'raw_data/links.csv', posters_root_dir = 'clean_data/downloaded_posters/poster/', posters_path = 'raw_data/movie_posters.csv', metadata_path = 'clean_data/metadata_IMDB.csv', ratings_path = 'raw_data/ratings.csv', batch_size=20, shuffle=False)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=48)
    test_loader = DataLoader(test_dataset, batch_size=48)

    num_users = len(np.unique(user_ratings['userId'])) + 1
    metadata_cols = list(metadata.columns)[1:]

    metadata_dim = len(metadata_cols)

    model = MultimodalModel(num_users=num_users, metadata_dim=metadata_dim, use_resnet = use_img, use_bert = use_text, use_metadata = use_metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    os.makedirs(f"{name}/charts", exist_ok=True)
    os.makedirs(f"{name}/trained_models", exist_ok=True)
    os.makedirs(f"{name}/predictions", exist_ok=True)

    EPOCHS = 5
    train_losses = []
    val_losses = []
    val_rmses = []
      
    start_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_train_loss = 0
      
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for batch in train_loop:
            optimizer.zero_grad()
            output = model(user_id=batch['user_id'].to(device), movie_id=batch['movie_id'].to(device), input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), image=batch['image'].to(device), metadata=batch['metadata'].to(device))
            loss = loss_fn(output, batch['rating'].to(device))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
      
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
      
        # 🔍 Validation
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                preds = model(user_id=batch['user_id'].to(device), movie_id=batch['movie_id'].to(device), input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), image=batch['image'].to(device), metadata=batch['metadata'].to(device))
                targets = batch['rating'].to(device)
                loss = loss_fn(preds, targets)
                total_val_loss += loss.item()

                all_preds.append(preds)
                all_targets.append(targets)

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # 🔢 Compute RMSE
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        val_rmse = rmse(all_preds, all_targets)
        val_rmses.append(val_rmse)

        # Save per-epoch predictions/targets
        torch.save({
            "predictions": all_preds.cpu(),
            "targets": all_targets.cpu()
        }, f"{name}/predictions/epoch_{epoch+1:02d}.pt")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{name}/trained_models/best_model.pt")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_rmses': val_rmses
        }, f"{name}/trained_models/last_checkpoint.pt")
      
        print(f"Epoch {epoch+1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}, val_rmse = {val_rmse:.4f}")

    torch.save(model.state_dict(), f"{name}/trained_models/final_model.pt")
    print(f"Lowest RMSE achieved: {min(val_rmses):.4f} at epoch {val_rmses.index(min(val_rmses)) + 1}")
    loss_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses)+1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'rmse': val_rmses
    })
    loss_df.to_csv(f"{name}/charts/loss_log.csv", index=False)
      
    # 📉 Plot MSE (Train + Validation)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train MSE Loss')
    plt.plot(val_losses, label='Validation MSE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training & Validation MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}/charts/mse_loss_plot.png")
    plt.show()
      
    # 📏 Plot RMSE
    plt.figure(figsize=(8, 5))
    plt.plot(val_rmses, label='Validation RMSE', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation RMSE Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}/charts/rmse_plot.png")
    plt.show()

if __name__ == '__main__':
    #run_training(name = 'itm', use_img = True, use_text = True, use_metadata = True)
    #run_training(name = 'it_', use_img = True, use_text = True, use_metadata = False)
    #run_training(name = 'i_m', use_img = True, use_text = False, use_metadata = True)
    #run_training(name = 'i__', use_img = True, use_text = False, use_metadata = False)
    run_training(name = '_tm', use_img = False, use_text = True, use_metadata = True)
    run_training(name = '_t_', use_img = False, use_text = True, use_metadata = False)
    run_training(name = '__m', use_img = False, use_text = False, use_metadata = True)
    #run_training(name = '___', use_img = False, use_text = False, use_metadata = False)
