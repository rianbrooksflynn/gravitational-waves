from dataclasses import dataclass, field
import torch
import numpy as np


def get_score_type_mapping():
    return {
        'mse_scores': 'mse',
        'cosine_scores': 'cosine', 
        'mahalanobis_time_scores': 'mahalanobis_time',
        'mahalanobis_stft_scores': 'mahalanobis_stft',
        'kl_scores': 'kl'
}


@dataclass
class LossHistory:
    total_losses: list = field(default_factory=list)
    rec_losses: list = field(default_factory=list)
    time_losses: list = field(default_factory=list)
    freq_losses: list = field(default_factory=list)
    contrastive_time_losses: list = field(default_factory=list)
    contrastive_stft_losses: list = field(default_factory=list)
    kl_losses: list = field(default_factory=list)

    def __str__(self):
        return (f"  Losses:\n"
                f"    Total Losses: {np.mean(self.total_losses):.5f}\n"
                f"    Rec Losses: {np.mean(self.rec_losses):.5f}\n"
                f"    Time Losses: {np.mean(self.time_losses):.5f}\n"
                f"    Freq Losses: {np.mean(self.freq_losses):.5f}\n"
                f"    Contrastive Time Losses: {np.mean(self.contrastive_time_losses):.5f}\n"
                f"    Contrastive Stft Losses: {np.mean(self.contrastive_stft_losses):.5f}\n"
                f"    Kl Losses: {np.mean(self.kl_losses):.5f}\n"
                f")")
    
    def add_epoch_losses(self, epoch_losses):
        self.total_losses.append(np.mean(epoch_losses.total_losses).item())
        self.rec_losses.append(np.mean(epoch_losses.rec_losses).item())
        self.time_losses.append(np.mean(epoch_losses.time_losses).item())
        self.freq_losses.append(np.mean(epoch_losses.freq_losses).item())
        self.contrastive_time_losses.append(np.mean(epoch_losses.contrastive_time_losses).item())
        self.contrastive_stft_losses.append(np.mean(epoch_losses.contrastive_stft_losses).item())
        self.kl_losses.append(np.mean(epoch_losses.kl_losses).item())


@dataclass 
class ScoreStatistics:
    mu: float = 0.0
    sigma: float = 0.0


@dataclass
class ScoreTypeStatistics:
    mse: ScoreStatistics = field(default_factory=ScoreStatistics)
    cosine: ScoreStatistics = field(default_factory=ScoreStatistics)
    mahalanobis_time: ScoreStatistics = field(default_factory=ScoreStatistics)
    mahalanobis_stft: ScoreStatistics = field(default_factory=ScoreStatistics)
    kl: ScoreStatistics = field(default_factory=ScoreStatistics)


@dataclass
class ScoreHistory:
    anomaly_scores: list = field(default_factory=list)
    mse_scores: list = field(default_factory=list)
    cosine_scores: list = field(default_factory=list)
    mahalanobis_time_scores: list = field(default_factory=list)
    mahalanobis_stft_scores: list = field(default_factory=list)
    kl_scores: list = field(default_factory=list)

    def __str__(self):
        return (f"  Scores:\n"
                f"    Anomaly Scores: {np.mean(self.anomaly_scores):.5f}\n"
                f"    MSE Scores: {np.mean(self.mse_scores):.5f}\n"
                f"    Cosine Scores: {np.mean(self.cosine_scores):.5f}\n"
                f"    Mahalanobis Time Scores: {np.mean(self.mahalanobis_time_scores):.5f}\n"
                f"    Mahalanobis Stft Scores: {np.mean(self.mahalanobis_stft_scores):.5f}\n"
                f"    Kl Scores: {np.mean(self.kl_scores):.5f}\n"
                f")")
    
    def add_epoch_scores(self, epoch_scores):
        self.anomaly_scores.append(np.mean(epoch_scores.anomaly_scores).item())
        self.mse_scores.append(np.mean(epoch_scores.mse_scores).item())
        self.cosine_scores.append(np.mean(epoch_scores.cosine_scores).item())
        self.mahalanobis_time_scores.append(np.mean(epoch_scores.mahalanobis_time_scores).item())
        self.mahalanobis_stft_scores.append(np.mean(epoch_scores.mahalanobis_stft_scores).item())
        self.kl_scores.append(np.mean(epoch_scores.kl_scores).item())



@dataclass
class MetricHistory:
    losses: LossHistory = field(default_factory=LossHistory)
    scores: ScoreHistory = field(default_factory=ScoreHistory)


@dataclass 
class ValidationMetricHistory(MetricHistory):
    auc_scores: list = field(default_factory=list)
    tnr: list = field(default_factory=list)


@dataclass
class TrainingHistory:
    training: MetricHistory = field(default_factory=MetricHistory)
    validation: ValidationMetricHistory = field(default_factory=ValidationMetricHistory)

    def __str__(self):
        return (f"Training Metrics:\n"
                f"{self.training.losses}\n"
                f"{self.training.scores}\n"
                f"Validation Metrics:\n"
                f"{self.validation.losses}\n"
                f"{self.validation.scores}\n"
                f"  AUC Scores: {self.validation.auc_scores}\n"
                f"  TNR: {self.validation.tnr}\n")
    
    def add_epoch_training_history(self, epoch_training_history):
        self.training.losses.add_epoch_losses(epoch_training_history.training.losses)
        self.training.scores.add_epoch_scores(epoch_training_history.training.scores)
        self.validation.losses.add_epoch_losses(epoch_training_history.validation.losses)
        self.validation.scores.add_epoch_scores(epoch_training_history.validation.scores)
        self.validation.auc_scores.append(epoch_training_history.validation.auc_scores[0])
        self.validation.tnr.append(epoch_training_history.validation.tnr[0])


def concatenate_tensors(tensor_lists):
    """Concatenate a list of tensors along the first dimension and convert to flattened numpy array."""
    return torch.cat(tensor_lists, dim=0).flatten().cpu().numpy()


def stack_and_multiply(scores, weights):
    """Stack scores and multiply by weights, then sum along the last dimension."""
    stacked_scores = np.stack(scores, axis=1)
    weighted_scores = stacked_scores * weights
    return np.sum(weighted_scores, axis=1)


def compute_score_statistics(scores, score_type_mapping):
    """Compute mean and std for each score type and return as ScoreTypeStatistics."""
    stats = ScoreTypeStatistics()
    
    for score_type, stat_field in score_type_mapping.items():
        score_values = getattr(scores, score_type)
        stat = ScoreStatistics(
            mu=np.mean(score_values).item(),
            sigma=np.std(score_values).item()
        )
        setattr(stats, stat_field, stat)
        
    return stats


def normalize_scores(score_history, score_statistics, score_type_mapping):
    """Normalize each score in ScoreHistory using the corresponding statistics from ScoreTypeStatistics."""
    normalized_scores = ScoreHistory()
    
    for score_type in score_type_mapping.keys():
        scores = torch.from_numpy(getattr(score_history, score_type))
        stat = getattr(score_statistics, score_type_mapping[score_type])
        normalized = (scores - stat.mu) / (stat.sigma + 1e-6)
        normalized = torch.sigmoid(normalized).cpu().numpy()
        setattr(normalized_scores, score_type, normalized)

    return normalized_scores


def contrastive_loss(features, class_labels, margin):
    batch_size, _, _ = features.shape
    flattened_features = features.view(batch_size, -1)

    pairwise_distances = torch.cdist(flattened_features, flattened_features, p=2)

    diag_mask = torch.eye(batch_size, device=features.device).bool()

    positive_mask = (class_labels.unsqueeze(1) == class_labels.unsqueeze(0))
    positive_mask &= ~diag_mask

    negative_mask = (class_labels.unsqueeze(1) != class_labels.unsqueeze(0))
    negative_mask &= ~diag_mask

    if positive_mask.sum() > 0:
        positive_loss = torch.exp(-pairwise_distances[positive_mask]).mean()
    else:
        positive_loss = torch.tensor(0.0, device=features.device)

    if negative_mask.sum() > 0:
        negative_loss = torch.clamp(margin - pairwise_distances[negative_mask], min=0).pow(2).mean()
    else:
        negative_loss = torch.tensor(0.0, device=features.device)

    return positive_loss + negative_loss


def compute_mahalanobis_params(z):
    mu = torch.mean(z, dim=0)
    cov = torch.cov(z.T)
    inv_cov = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0]).to(z.device))
    return mu, inv_cov


def mahalanobis_distance(z, mu, inv_cov):
    delta = z - mu
    return torch.sqrt(torch.clamp(torch.sum(delta * torch.matmul(delta, inv_cov), dim=1), min=1e-6))


def kl_warmup(epoch, max_epochs, max_kl_weight):
    """Gradually increase KL loss weight over training."""
    return min(1.0, epoch / (max_epochs * 0.2)) * max_kl_weight
