from final_model import DualStreamModel
from utils import (
    concatenate_tensors,
    contrastive_loss,
    compute_mahalanobis_params,
    compute_score_statistics,
    get_score_type_mapping,
    kl_warmup,
    mahalanobis_distance,
    normalize_scores,
    stack_and_multiply,
    TrainingHistory
)
from dataclasses import asdict
import json
import numpy as np
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def train_model(
        epochs,
        batch_size,
        lr,
        weight_decay,
        embed_dim, 
        num_heads, 
        hidden_dim_transformer, 
        dropout_transformer, 
        num_transformer_layers, 
        latent_dim_transformer, 
        hidden_dim_cnn, 
        latent_dim_cnn,
        split,
        loss_weights,
        max_kl_weight,
        anomaly_weights
):
    seq_len = 200
    feature_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualStreamModel(seq_len, feature_dim, embed_dim, num_heads, hidden_dim_transformer, dropout_transformer, num_transformer_layers, latent_dim_transformer, hidden_dim_cnn, latent_dim_cnn)
    model.to(device)

    # Set up datasets
    background = np.load(pathlib.Path(__file__).parent.parent / "datasets/background.npz")["data"]
    bbh = np.load(pathlib.Path(__file__).parent.parent / "datasets/bbh_for_challenge.npy")
    sglf = np.load(pathlib.Path(__file__).parent.parent / "datasets/sglf_for_challenge.npy")

    # Normalize and prepare data
    background_stds = np.std(background, axis=-1)[:, :, np.newaxis]
    bbh_stds = np.std(bbh, axis=-1)[:, :, np.newaxis]
    sglf_stds = np.std(sglf, axis=-1)[:, :, np.newaxis]
    background = background / background_stds
    bbh = bbh / bbh_stds
    sglf = sglf / sglf_stds
    background = torch.from_numpy(background).float().to(device)
    bbh = torch.from_numpy(bbh).float().to(device)
    sglf = torch.from_numpy(sglf).float().to(device)
    background = torch.swapaxes(background, 1, 2)
    bbh = torch.swapaxes(bbh, 1, 2)
    sglf = torch.swapaxes(sglf, 1, 2)

    # Train/test split
    background_train, background_test = train_test_split(background, test_size=split, random_state=42)
    bbh_train, bbh_test = train_test_split(bbh, test_size=split, random_state=42)
    num_train_batches = (background_train.shape[0] + bbh_train.shape[0]) // batch_size
    batch_size_per_class = batch_size // 2

    # Validation set
    val_data = torch.cat([background_test, bbh_test, sglf], dim=0)
    val_data_class_labels = torch.cat([torch.zeros(background_test.shape[0]), torch.ones(bbh_test.shape[0]), 2 * torch.ones(sglf.shape[0])], dim=0).to(device)
    val_data_labels = torch.cat([torch.zeros(background_test.shape[0]), torch.zeros(bbh_test.shape[0]), torch.ones(sglf.shape[0])], dim=0).to(device)
    val_data_shuffled_indices = torch.randperm(val_data.shape[0])
    val_data = val_data[val_data_shuffled_indices]
    val_data_labels = val_data_labels[val_data_shuffled_indices]
    num_val_batches = val_data.shape[0] // batch_size

    # Optimizer and scheduler
    optimizer_model = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=epochs)
    rec_loss = nn.MSELoss().to(device)

    training_history = TrainingHistory()
    best_auc_score = 0.0
    best_tnr = 0.0

    # Training loop
    for i in tqdm(range(epochs)):
        background_train_shuffle_indices = torch.randperm(background_train.shape[0])
        background_train = background_train[background_train_shuffle_indices]
        bbh_train_shuffle_indices = torch.randperm(bbh_train.shape[0])
        bbh_train = bbh_train[bbh_train_shuffle_indices]

        epoch_training_history = TrainingHistory()

        z_train_time_background = []
        z_train_stft_background = []
        z_train_time_bbh = []
        z_train_stft_bbh = []

        model.train()
        for batch in range(num_train_batches):
            x_background = background_train[batch * batch_size_per_class:(batch + 1) * batch_size_per_class]
            x_bbh = bbh_train[batch * batch_size_per_class:(batch + 1) * batch_size_per_class]
            x = torch.cat([x_background, x_bbh], dim=0)
            x_labels = torch.cat([torch.zeros(x_background.shape[0]), torch.ones(x_bbh.shape[0])], dim=0).to(device)
            x_labels_shuffle_indices = torch.randperm(x_labels.shape[0]).to(device)
            x = x[x_labels_shuffle_indices]
            x_labels = x_labels[x_labels_shuffle_indices]

            optimizer_model.zero_grad()
            y, time_reconstruction, freq_reconstruction, time_encoded, stft_encoded, kl_loss, time_mu, stft_mu = model(x)
            loss_rec = rec_loss(y, x)
            loss_time = rec_loss(time_reconstruction, x)
            loss_freq = rec_loss(freq_reconstruction, x)
            loss_contrastive_time = contrastive_loss(time_encoded, x_labels, margin=1.0)
            loss_contrastive_stft = contrastive_loss(stft_encoded, x_labels, margin=1.0)
            loss_kl = kl_loss.mean()

            loss_weights[5] = kl_warmup(i+1, epochs, max_kl_weight)
            loss = map(lambda x, y: x * y, loss_weights, [loss_rec, loss_time, loss_freq, loss_contrastive_time, loss_contrastive_stft, loss_kl])
            loss = sum(loss)
            loss.backward()
            optimizer_model.step()

            time_mu_background = time_mu[~x_labels.bool()]
            stft_mu_background = stft_mu[~x_labels.bool()]
            time_mu_bbh = time_mu[x_labels.bool()]
            stft_mu_bbh = stft_mu[x_labels.bool()]

            epoch_training_history.training.losses.total_losses.append(loss.item())
            epoch_training_history.training.losses.rec_losses.append(loss_rec.item())
            epoch_training_history.training.losses.time_losses.append(loss_time.item())
            epoch_training_history.training.losses.freq_losses.append(loss_freq.item())
            epoch_training_history.training.losses.contrastive_time_losses.append(loss_contrastive_time.item())
            epoch_training_history.training.losses.contrastive_stft_losses.append(loss_contrastive_stft.item())
            epoch_training_history.training.losses.kl_losses.append(loss_kl.item())
            z_train_time_background.append(time_mu_background.detach())
            z_train_stft_background.append(stft_mu_background.detach())
            z_train_time_bbh.append(time_mu_bbh.detach())
            z_train_stft_bbh.append(stft_mu_bbh.detach())

        scheduler.step()

        model.eval()

        # Go back through training data and compute anomaly scores
        with torch.no_grad():
            z_train_time_background = torch.cat(z_train_time_background, dim=0)
            z_train_stft_background = torch.cat(z_train_stft_background, dim=0)
            z_train_time_bbh = torch.cat(z_train_time_bbh, dim=0)
            z_train_stft_bbh = torch.cat(z_train_stft_bbh, dim=0)

            mu_train_time_background, inv_cov_train_time_background = compute_mahalanobis_params(z_train_time_background)
            mu_train_stft_background, inv_cov_train_stft_background = compute_mahalanobis_params(z_train_stft_background)
            mu_train_time_bbh, inv_cov_train_time_bbh = compute_mahalanobis_params(z_train_time_bbh)
            mu_train_stft_bbh, inv_cov_train_stft_bbh = compute_mahalanobis_params(z_train_stft_bbh)

            for batch in range(num_train_batches):
                x_background = background_train[batch * batch_size_per_class:(batch + 1) * batch_size_per_class]
                x_bbh = bbh_train[batch * batch_size_per_class:(batch + 1) * batch_size_per_class]
                x = torch.cat([x_background, x_bbh], dim=0)
                y, _, _, _, _, kl_loss, time_mu, stft_mu = model(x)

                epoch_training_history.training.scores.mse_scores.append(torch.mean((x - y) ** 2, dim=(1, 2)))
                epoch_training_history.training.scores.cosine_scores.append(1 - F.cosine_similarity(x, y, dim=-1).mean(dim=-1))
                epoch_training_history.training.scores.mahalanobis_time_scores.append(mahalanobis_distance(time_mu, mu_train_time_background, inv_cov_train_time_background) + mahalanobis_distance(time_mu, mu_train_time_bbh, inv_cov_train_time_bbh))
                epoch_training_history.training.scores.mahalanobis_stft_scores.append(mahalanobis_distance(stft_mu, mu_train_stft_background, inv_cov_train_stft_background) + mahalanobis_distance(stft_mu, mu_train_stft_bbh, inv_cov_train_stft_bbh))
                epoch_training_history.training.scores.kl_scores.append(kl_loss)

            # Concatenate all scores from training batches
            score_type_mapping = get_score_type_mapping()
            for score_type in score_type_mapping.keys():
                setattr(epoch_training_history.training.scores, score_type, 
                       concatenate_tensors(getattr(epoch_training_history.training.scores, score_type)))

            statistics = compute_score_statistics(epoch_training_history.training.scores, score_type_mapping)

            epoch_training_history.training.scores = normalize_scores(epoch_training_history.training.scores, statistics, score_type_mapping)

            epoch_training_history.training.scores.anomaly_scores = stack_and_multiply(
                [getattr(epoch_training_history.training.scores, score_type) for score_type in score_type_mapping.keys()],
                anomaly_weights
            )

            for batch in range(num_val_batches):
                x = val_data[batch * batch_size:(batch + 1) * batch_size]
                x_labels = val_data_labels[batch * batch_size:(batch + 1) * batch_size]
                x_class_labels = val_data_class_labels[batch * batch_size:(batch + 1) * batch_size]
                
                y, time_reconstruction, freq_reconstruction, time_encoded, stft_encoded, kl_loss, time_mu, stft_mu = model(x)
                loss_rec = rec_loss(y, x)
                loss_time = rec_loss(time_reconstruction, x)
                loss_freq = rec_loss(freq_reconstruction, x)
                loss_contrastive_time = contrastive_loss(time_encoded, x_class_labels, margin=1.0)
                loss_contrastive_stft = contrastive_loss(stft_encoded, x_class_labels, margin=1.0)
                loss_kl = kl_loss.mean()

                loss_weights[5] = kl_warmup(i+1, epochs, max_kl_weight)
                loss = map(lambda x, y: x * y, loss_weights, [loss_rec, loss_time, loss_freq, loss_contrastive_time, loss_contrastive_stft, loss_kl])
                loss = sum(loss)

                epoch_training_history.validation.losses.total_losses.append(loss.item())
                epoch_training_history.validation.losses.rec_losses.append(loss_rec.item())
                epoch_training_history.validation.losses.time_losses.append(loss_time.item())
                epoch_training_history.validation.losses.freq_losses.append(loss_freq.item())
                epoch_training_history.validation.losses.contrastive_time_losses.append(loss_contrastive_time.item())
                epoch_training_history.validation.losses.contrastive_stft_losses.append(loss_contrastive_stft.item())
                epoch_training_history.validation.losses.kl_losses.append(loss_kl.item())

                epoch_training_history.validation.scores.mse_scores.append(torch.mean((x - y) ** 2, dim=(1, 2)))
                epoch_training_history.validation.scores.cosine_scores.append(1 - F.cosine_similarity(x, y, dim=-1).mean(dim=-1))
                epoch_training_history.validation.scores.mahalanobis_time_scores.append(mahalanobis_distance(time_mu, mu_train_time_background, inv_cov_train_time_background) + mahalanobis_distance(time_mu, mu_train_time_bbh, inv_cov_train_time_bbh))
                epoch_training_history.validation.scores.mahalanobis_stft_scores.append(mahalanobis_distance(stft_mu, mu_train_stft_background, inv_cov_train_stft_background) + mahalanobis_distance(stft_mu, mu_train_stft_bbh, inv_cov_train_stft_bbh))
                epoch_training_history.validation.scores.kl_scores.append(kl_loss)

            # Concatenate all scores from validation batches
            for score_type in score_type_mapping.keys():
                setattr(epoch_training_history.validation.scores, score_type, 
                       concatenate_tensors(getattr(epoch_training_history.validation.scores, score_type)))
            
            epoch_training_history.validation.scores = normalize_scores(epoch_training_history.validation.scores, statistics, score_type_mapping)
            
            epoch_training_history.validation.scores.anomaly_scores = stack_and_multiply(
                [getattr(epoch_training_history.validation.scores, score_type) for score_type in score_type_mapping.keys()],
                anomaly_weights
            )

            auc_score = roc_auc_score(val_data_labels.cpu().numpy(), epoch_training_history.validation.scores.anomaly_scores)
            fpr, tpr, _ = roc_curve(val_data_labels.cpu().numpy(), epoch_training_history.validation.scores.anomaly_scores)
            tnr = 1 - fpr[np.argmax(tpr >= 0.9)]

            epoch_training_history.validation.auc_scores = [auc_score]
            epoch_training_history.validation.tnr = [tnr]

            tqdm.write(f"Epoch {i+1}/{epochs} \n"
                       f"{epoch_training_history}")

            training_history.add_epoch_training_history(epoch_training_history)

            if auc_score > best_auc_score:
                best_auc_score = auc_score
                torch.save(model.state_dict(), pathlib.Path(__file__).parent / f"best_auc_model_weights.pt")
            
            if tnr > best_tnr:
                best_tnr = tnr
                torch.save(model.state_dict(), pathlib.Path(__file__).parent / f"best_tnr_model_weights.pt")

            with open(pathlib.Path(__file__).parent / f"training_history.json", "w") as f:
                json.dump(asdict(training_history), f)
            
    torch.save(model.state_dict(), pathlib.Path(__file__).parent / f"final_model_weights.pt")


def main():
    loss_weights = [0.8, 0.85, 0.1, 0.55, 0.5, 0.02]
    loss_weights = [w / sum(loss_weights) for w in loss_weights]
    anomaly_weights = [0.0, 0.0, 0.0, 0.0, 1.0]
    anomaly_weights = [w / sum(anomaly_weights) for w in anomaly_weights]
    train_model(
        epochs=25,
        batch_size=100,
        lr=1e-6,
        weight_decay=1e-5,
        embed_dim=8 * 8,
        num_heads=8,
        hidden_dim_transformer=64,
        dropout_transformer=0.2,
        num_transformer_layers=6,
        latent_dim_transformer=2 * 8,
        hidden_dim_cnn=64,
        latent_dim_cnn=32,
        split=0.25,
        loss_weights=loss_weights,
        max_kl_weight=0.05,
        anomaly_weights=anomaly_weights
    )


if __name__ == "__main__":
    main()
