from final_model import DualStreamModel
from utils import (
    compute_mahalanobis_params,
    mahalanobis_distance
)
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from matplotlib import pyplot as plt


def test_model():
    seq_len = 200
    feature_dim = 2
    batch_size = 20
    embed_dim = 8 * 8
    num_heads = 8
    hidden_dim_transformer = 64
    dropout_transformer = 0.2
    num_transformer_layers = 6
    latent_dim_transformer = 2 * 8
    hidden_dim_cnn = 64
    latent_dim_cnn = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = DualStreamModel(seq_len, feature_dim, embed_dim, num_heads, hidden_dim_transformer, dropout_transformer, num_transformer_layers, latent_dim_transformer, hidden_dim_cnn, latent_dim_cnn)
    model.load_state_dict(torch.load(pathlib.Path(__file__).parent / "best_tnr_model_weights.pt"))
    model.to(device)
    model.eval()

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

    # Normalization dataset
    norm_data = torch.cat([background, bbh], dim=0)
    norm_data_labels = torch.cat([torch.zeros(background.shape[0]), torch.ones(bbh.shape[0])], dim=0).to(device)
    num_norm_batches = norm_data.shape[0] // batch_size

    # Testing dataset
    test_data = torch.cat([background, bbh, sglf], dim=0)
    test_data_labels = torch.cat([torch.zeros(background.shape[0]), torch.zeros(bbh.shape[0]), torch.ones(sglf.shape[0])], dim=0).to(device)
    num_test_batches = test_data.shape[0] // batch_size

    with torch.no_grad():
        # Finding Mahalanobis parameters
        z_norm_time_background = []
        z_norm_stft_background = []
        z_norm_time_bbh = []
        z_norm_stft_bbh = []
        for batch in range(num_norm_batches):
            x = norm_data[batch * batch_size:(batch + 1) * batch_size]
            x_labels = norm_data_labels[batch * batch_size:(batch + 1) * batch_size]

            y, _, _, _, _, kl_loss, time_mu, stft_mu = model(x)

            z_norm_time_background.append(time_mu[~x_labels.bool()])
            z_norm_stft_background.append(stft_mu[~x_labels.bool()])
            z_norm_time_bbh.append(time_mu[x_labels.bool()])
            z_norm_stft_bbh.append(stft_mu[x_labels.bool()])

        z_norm_time_background = torch.cat(z_norm_time_background, dim=0)
        z_norm_stft_background = torch.cat(z_norm_stft_background, dim=0)
        z_norm_time_bbh = torch.cat(z_norm_time_bbh, dim=0)
        z_norm_stft_bbh = torch.cat(z_norm_stft_bbh, dim=0)

        mu_norm_time_background, inv_cov_norm_time_background = compute_mahalanobis_params(z_norm_time_background)
        mu_norm_stft_background, inv_cov_norm_stft_background = compute_mahalanobis_params(z_norm_stft_background)
        mu_norm_time_bbh, inv_cov_norm_time_bbh = compute_mahalanobis_params(z_norm_time_bbh)
        mu_norm_stft_bbh, inv_cov_norm_stft_bbh = compute_mahalanobis_params(z_norm_stft_bbh)

        # Save Mahalanobis parameters
        mahalanobis_params = {
            "mu_norm_time_background": mu_norm_time_background,
            "inv_cov_norm_time_background": inv_cov_norm_time_background,
            "mu_norm_stft_background": mu_norm_stft_background,
            "inv_cov_norm_stft_background": inv_cov_norm_stft_background,
            "mu_norm_time_bbh": mu_norm_time_bbh,
            "inv_cov_norm_time_bbh": inv_cov_norm_time_bbh,
            "mu_norm_stft_bbh": mu_norm_stft_bbh,
            "inv_cov_norm_stft_bbh": inv_cov_norm_stft_bbh
        }
        torch.save(mahalanobis_params, pathlib.Path(__file__).parent / "mahalanobis_params.pt")

        # Normalizing anomaly scores
        norm_mse_scores = []
        norm_cosine_scores = []
        norm_mahalanobis_time_scores = []
        norm_mahalanobis_stft_scores = []
        norm_kl_scores = []
        for batch in range(num_norm_batches):
            x = norm_data[batch * batch_size:(batch + 1) * batch_size]
            x_labels = norm_data_labels[batch * batch_size:(batch + 1) * batch_size]

            y, _, _, _, _, kl_loss, time_mu, stft_mu = model(x)

            norm_mse_scores.append(torch.mean((x - y) ** 2, dim=(1, 2)))
            norm_cosine_scores.append(1 - F.cosine_similarity(x, y, dim=-1).mean(dim=-1))
            norm_mahalanobis_time_scores.append(mahalanobis_distance(time_mu, mu_norm_time_background, inv_cov_norm_time_background) + mahalanobis_distance(time_mu, mu_norm_time_bbh, inv_cov_norm_time_bbh))
            norm_mahalanobis_stft_scores.append(mahalanobis_distance(stft_mu, mu_norm_stft_background, inv_cov_norm_stft_background) + mahalanobis_distance(stft_mu, mu_norm_stft_bbh, inv_cov_norm_stft_bbh))
            norm_kl_scores.append(kl_loss)

        norm_mse_scores = torch.cat(norm_mse_scores, dim=0)
        norm_cosine_scores = torch.cat(norm_cosine_scores, dim=0)
        norm_mahalanobis_time_scores = torch.cat(norm_mahalanobis_time_scores, dim=0)
        norm_mahalanobis_stft_scores = torch.cat(norm_mahalanobis_stft_scores, dim=0)
        norm_kl_scores = torch.cat(norm_kl_scores, dim=0)

        mse_mean = torch.mean(norm_mse_scores)
        cosine_mean = torch.mean(norm_cosine_scores)
        mahalanobis_time_mean = torch.mean(norm_mahalanobis_time_scores)
        mahalanobis_stft_mean = torch.mean(norm_mahalanobis_stft_scores)
        kl_mean = torch.mean(norm_kl_scores)

        mse_std = torch.std(norm_mse_scores)
        cosine_std = torch.std(norm_cosine_scores)
        mahalanobis_time_std = torch.std(norm_mahalanobis_time_scores)
        mahalanobis_stft_std = torch.std(norm_mahalanobis_stft_scores)
        kl_std = torch.std(norm_kl_scores)

        # Testing loop
        test_mse_scores = []
        test_cosine_scores = []
        test_mahalanobis_time_scores = []
        test_mahalanobis_stft_scores = []
        test_kl_scores = []

        for batch in range(num_test_batches):
            x = test_data[batch * batch_size:(batch + 1) * batch_size]
            x_labels = test_data_labels[batch * batch_size:(batch + 1) * batch_size]

            y, _, _, _, _, kl_loss, time_mu, stft_mu = model(x)

            test_mse_scores.append(torch.mean((x - y) ** 2, dim=(1, 2)))
            test_cosine_scores.append(1 - F.cosine_similarity(x, y, dim=-1).mean(dim=-1))
            test_mahalanobis_time_scores.append(mahalanobis_distance(time_mu, mu_norm_time_background, inv_cov_norm_time_background) + mahalanobis_distance(time_mu, mu_norm_time_bbh, inv_cov_norm_time_bbh))
            test_mahalanobis_stft_scores.append(mahalanobis_distance(stft_mu, mu_norm_stft_background, inv_cov_norm_stft_background) + mahalanobis_distance(stft_mu, mu_norm_stft_bbh, inv_cov_norm_stft_bbh))
            test_kl_scores.append(kl_loss)

        test_mse_scores = torch.cat(test_mse_scores, dim=0)
        test_cosine_scores = torch.cat(test_cosine_scores, dim=0)
        test_mahalanobis_time_scores = torch.cat(test_mahalanobis_time_scores, dim=0)
        test_mahalanobis_stft_scores = torch.cat(test_mahalanobis_stft_scores, dim=0)
        test_kl_scores = torch.cat(test_kl_scores, dim=0)

        test_mse_scores = (test_mse_scores - mse_mean) / (mse_std + 1e-6)
        test_cosine_scores = (test_cosine_scores - cosine_mean) / (cosine_std + 1e-6)
        test_mahalanobis_time_scores = (test_mahalanobis_time_scores - mahalanobis_time_mean) / (mahalanobis_time_std + 1e-6)
        test_mahalanobis_stft_scores = (test_mahalanobis_stft_scores - mahalanobis_stft_mean) / (mahalanobis_stft_std + 1e-6)
        test_kl_scores = (test_kl_scores - kl_mean) / (kl_std + 1e-6)

        test_mse_scores = torch.sigmoid(test_mse_scores).cpu().numpy()
        test_cosine_scores = torch.sigmoid(test_cosine_scores).cpu().numpy()
        test_mahalanobis_time_scores = torch.sigmoid(test_mahalanobis_time_scores).cpu().numpy()
        test_mahalanobis_stft_scores = torch.sigmoid(test_mahalanobis_stft_scores).cpu().numpy()
        test_kl_scores = torch.sigmoid(test_kl_scores).cpu().numpy()

        # Save kl score mean and std
        kl_score_mean_std = {
            "kl_mean": kl_mean,
            "kl_std": kl_std
        }
        torch.save(kl_score_mean_std, pathlib.Path(__file__).parent / "kl_score_mean_std.pt")

        labels = test_data_labels.cpu().numpy()
        scores = test_kl_scores
        fpr, tpr, _ = roc_curve(labels, scores)
        tnr = 1 - fpr[np.argmax(tpr >= 0.9)]
        print(f"TNR: {tnr}")

        # Plot ROC curve
        roc = roc_auc_score(labels, scores)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc:.3f})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(pathlib.Path(__file__).parent / "final_roc_curve.png")
        plt.close()


if __name__ == "__main__":
    test_model()
