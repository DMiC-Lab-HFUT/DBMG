import torch
import torch.nn.functional as F


class AdaptiveLossWeighting(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.log_sigma1 = torch.nn.Parameter(torch.tensor(0.0))  #
        self.log_sigma2 = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma3 = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, l1, l2, l3):
        w1 = 1 / (2 * torch.exp(self.log_sigma1)**2)
        w2 = 1 / (2 * torch.exp(self.log_sigma2)**2)
        w3 = 1 / (2 * torch.exp(self.log_sigma3)**2)
        return (w1 * l1 + self.log_sigma1 + 
                w2 * l2 + self.log_sigma2 + 
                w3 * l3 + self.log_sigma3)


def category_softmax_loss(similarity_matrix, text_labels, image_labels, tau=0.05):
    B = similarity_matrix.size(0)
    device = similarity_matrix.device
    sim = similarity_matrix / tau
    labels = (text_labels.unsqueeze(1) == image_labels.unsqueeze(0)).float().to(device)  # [B,B]

    loss_row = - (labels * F.log_softmax(sim, dim=1)).sum(dim=1) / labels.sum(dim=1).clamp(min=1)
    loss_col = - (labels.T * F.log_softmax(sim.T, dim=1)).sum(dim=1) / labels.T.sum(dim=1).clamp(min=1)
    return (loss_row.mean() + loss_col.mean()) / 2


def pairwise_ranking_loss(similarity_matrix, labels, margin=0.35):
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(similarity_matrix.device)
    loss = 0.0
    count = 0
    for i in range(similarity_matrix.size(0)):
        pos_scores = similarity_matrix[i][pos_mask[i]]
        neg_scores = similarity_matrix[i][~pos_mask[i]]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            continue
        pairwise_diff = margin + neg_scores.unsqueeze(0) - pos_scores.unsqueeze(1)
        loss += F.relu(pairwise_diff).mean()
        count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=similarity_matrix.device, requires_grad=True)
