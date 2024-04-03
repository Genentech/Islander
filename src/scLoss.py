import torch, torch.nn as nn, torch.nn.functional as F, Utils_Handler as uh


class HSICLoss(nn.Module):
    """Hilbert-Schmidt Independence Criterion (HSIC) loss function"""

    def __init__(self, s_x=1, s_y=1):
        super(HSICLoss, self).__init__()
        self.s_x = s_x
        self.s_y = s_y

    @staticmethod
    def pairwise_distances(x):
        instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    @staticmethod
    def GaussianKernelMatrix(x, sigma=1.0):
        pairwise_distances_ = HSICLoss.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ / sigma)

    def forward(self, x, y):
        m, _ = x.shape  # batch size
        K = self.GaussianKernelMatrix(x, self.s_x)
        L = self.GaussianKernelMatrix(y, self.s_y)
        H = torch.eye(m, device=x.device) - 1.0 / m * torch.ones(
            (m, m), device=x.device
        )
        # H = H.to(x.device)
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC


# class HSICLoss(nn.Module):
#     """Hilbert-Schmidt Independence Criterion (HSIC) loss function"""

#     def __init__(self, s_x=1, s_y=1):
#         super(HSICLoss, self).__init__()
#         self.s_x = s_x
#         self.s_y = s_y

#     @staticmethod
#     def GaussianKernelMatrix(x, sigma=1.0):
#         # Exploit broadcasting
#         x_norm = (x**2).sum(dim=1).view(-1, 1)
#         pairwise_distances_ = x_norm - 2.0 * x @ x.t() + x_norm.t()
#         # return torch.exp(-pairwise_distances_ / (2 * sigma**2))
#         return torch.exp(-pairwise_distances_ / sigma)

#     def forward(self, x, y):
#         m, _ = x.shape  # batch size
#         K = self.GaussianKernelMatrix(x, self.s_x)
#         L = self.GaussianKernelMatrix(y, self.s_y)

#         # Using broadcasting and in-place operations
#         H = torch.eye(m, device=x.device).sub_(1.0 / m)

#         HKH = torch.mm(K, H)
#         LHKH = torch.mm(L, HKH)

#         # Exploit trace of matrix multiplication properties: trace(AB) = sum(A * B^T)
#         HSIC = (LHKH * H).sum() / ((m - 1) ** 2)
#         return HSIC


class DDCLoss(nn.Module):
    """Differentiable Distance Correlation (DDC) loss function"""

    def __init__(self):
        super(DDCLoss, self).__init__()

    def forward(self, x, y):
        def distance_matrix(data):
            square = torch.square(data)
            sum_square = torch.sum(square, dim=-1, keepdim=True)
            distances = sum_square + sum_square.t() - 2 * torch.matmul(data, data.t())
            return torch.sqrt(torch.clamp(distances, min=1e-12))

        def centering(matrix):
            # n = matrix.size(0)
            mean_rows = torch.mean(matrix, dim=0, keepdim=True)
            mean_cols = torch.mean(matrix, dim=1, keepdim=True)
            mean_total = torch.mean(matrix)
            return matrix - mean_rows - mean_cols + mean_total

        dist_x, dist_y = distance_matrix(x), distance_matrix(y)
        A, B = centering(dist_x), centering(dist_y)
        dcor = torch.sqrt(torch.sum(A * B) / x.size(0) ** 2)
        return dcor


class KCCALoss(nn.Module):
    """eigenvalues, _ = torch.linalg.eig(M)
    kcca_corr = torch.sum(torch.sqrt(torch.clamp(eigenvalues.real, min=0))) (KCCA) loss function
    """

    def __init__(self, kernel="linear", sigma=None, reg=1e-5):
        super(KCCALoss, self).__init__()
        self.kernel = kernel
        self.sigma = sigma
        self.reg = reg

    def forward(self, x, y):
        def kernel_matrix(data, kernel="linear", sigma=None):
            if kernel == "linear":
                return torch.matmul(data, data.t())
            elif kernel == "rbf":
                square = torch.square(data)
                sum_square = torch.sum(square, dim=-1, keepdim=True)
                distances = (
                    sum_square + sum_square.t() - 2 * torch.matmul(data, data.t())
                )
                return torch.exp(-distances / (2 * sigma**2))
            else:
                raise ValueError("Invalid kernel type.")

        Kx = kernel_matrix(x, kernel=self.kernel, sigma=self.sigma)
        Ky = kernel_matrix(y, kernel=self.kernel, sigma=self.sigma)

        n = x.size(0)
        I = torch.eye(n).to(x.device)

        L, R = torch.inverse(Kx + self.reg * I), torch.inverse(Ky + self.reg * I)
        L_inv, R_inv = torch.inverse(L), torch.inverse(R)

        M = torch.matmul(L_inv, torch.matmul(Kx, torch.matmul(Ky, R_inv)))

        eigenvalues, _ = torch.linalg.eig(M)
        kcca_corr = torch.sum(torch.sqrt(torch.clamp(eigenvalues.real, min=0)))

        return kcca_corr


class InfoNCELoss(nn.Module):
    """Info Noise Contrastive Estimation Loss"""

    def __init__(self, queries, keys, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        query_dim = queries.shape[1]
        key_dim = keys.shape[1]
        device = keys.device
        if query_dim != key_dim:
            self.proj = nn.Linear(query_dim, key_dim).to(device)
        else:
            self.proj = None
        self.temperature = temperature
        self.queries = queries
        self.keys = keys

    def forward(self):
        queries = self.queries
        keys = self.keys
        if self.proj:
            queries = self.proj(queries)
        batch_size = queries.size(0)

        logits = torch.matmul(queries, keys.t()) / self.temperature
        labels = torch.arange(batch_size).to(logits.device)
        return nn.CrossEntropyLoss()(logits, labels)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        l2 = (emb1 - emb2).pow(2).sum(1)
        loss = torch.mean(
            (1 - label) * torch.pow(l2, 2)
            + (label) * torch.pow(torch.clamp(self.margin - l2, min=0.0), 2)
        )
        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeds, labels):
        # normalize the embeddings along the embedding dimension
        embeds = F.normalize(embeds, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeds, embeds.t())

        # Create mask for positive samples
        labels = labels.unsqueeze(1)
        positive_mask = torch.eq(labels, labels.t()).float()

        # Extract positive and negative similarities
        positive_sim = sim_matrix * positive_mask
        negative_sim = sim_matrix * (1 - positive_mask)

        # Compute logit (numerator) and sum of exp(logits) (denominator) for the loss function
        numerator = torch.sum(F.exp(positive_sim / self.temperature), dim=-1)
        denominator = torch.sum(F.exp(negative_sim / self.temperature), dim=-1)

        # Compute the loss
        loss = -torch.log(numerator / (denominator + 1e-7))
        return torch.mean(loss)


class DualBatchSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(DualBatchSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, emb_1, emb_2, labels1, labels2):
        """
        Args:
            emb_1, emb_2: Tensors of shape (batch_size, embedding_dim)
            labels1, labels2: Tensors of shape (batch_size)
        """

        # Normalize the embeddings
        emb_1 = F.normalize(emb_1, p=2, dim=-1)
        emb_2 = F.normalize(emb_2, p=2, dim=-1)

        # Concatenate the embeddings and labels
        embs = torch.cat([emb_1, emb_2], dim=0)
        labels = torch.cat([labels1, labels2], dim=0)

        # Compute the dot products
        dot_products = torch.mm(embs, embs.t()) / self.temperature

        # Compute the similarity scores
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
        labels_matrix = labels_matrix.float()

        # Subtract the similarity scores for positive pairs
        positive_pairs = dot_products * labels_matrix

        # Subtract the similarity scores for negative pairs
        negative_pairs = dot_products * (1 - labels_matrix)

        # Subtract the maximum similarity score for stability
        max_positive_pairs = positive_pairs.max(dim=1, keepdim=True)[0]
        max_negative_pairs = negative_pairs.max(dim=1, keepdim=True)[0]

        # Compute the logits
        logits = positive_pairs - max_positive_pairs - max_negative_pairs

        # Compute the log-sum-exp for the denominator
        logsumexp = torch.logsumexp(
            negative_pairs - max_negative_pairs, dim=1, keepdim=True
        )

        # Compute the loss
        loss = -logits + logsumexp

        return loss.mean()


class SCL(nn.Module):
    def __init__(self, memory_bank, temp=1, alpha=1e-4):
        super().__init__()
        self.temp = temp
        self.alpha = alpha  # 0.0001
        self.memory_bank = memory_bank

    def forward(self, embeds, labels, embeds_ema):
        embeds_bank, labels_bank = self.memory_bank.fetch(embeds_ema, labels)

        # mask
        mask_p = labels.view(-1, 1).eq(labels_bank.view(1, -1))
        mask_n = mask_p.logical_not()
        mask_p[:, : embeds.size(0)].fill_diagonal_(False)

        # logit
        x_norm = F.normalize(embeds, dim=1)
        f_norm = F.normalize(embeds_bank, dim=1)
        logits = x_norm.mm(f_norm.t()) / self.temp

        logits_p = torch.masked_select(logits, mask_p)
        logits_n = torch.masked_select(logits, mask_n)

        loss_p = F.binary_cross_entropy_with_logits(
            logits_p,
            torch.ones_like(logits_p),
        )
        loss_n = F.binary_cross_entropy_with_logits(
            logits_n,
            torch.zeros_like(logits_n),
        )
        loss = self.alpha * loss_p + (1.0 - self.alpha) * loss_n

        return loss


class SimPLE(nn.Module):
    def __init__(
        self,
        memory_bank,
        r=1.0,
        m=0.0,
        b_cos=0.2,
        lw=1000.0,
        alpha=1e-4,  # have to set this small
        b_logit=-10.0,
    ):
        """Simple Pairwise Similarity Learning with Memory Bank"""

        super().__init__()
        self.memory_bank = memory_bank
        self.rank = 0

        # === hyperparam ===
        # alpha is the lambda in paper Eqn. 5
        self.alpha = alpha
        self.b_cos = b_cos
        self.lw = lw
        self.r = r
        self.m = m
        #
        # init bias
        self.b_logit = nn.Parameter(b_logit + 0.0 * torch.Tensor(1)).to(
            self.memory_bank.embed_bank.device
        )

    # embeds, labels, embeds_ema,
    def forward(self, x, y, x_ema):
        # update bank
        self.feat_bank, self.label_bank = self.memory_bank.fetch(x_ema, y)

        # mask
        mask_p = y.view(-1, 1).eq(self.label_bank.view(1, -1))
        mask_n = mask_p.logical_not()
        pt = self.rank * x.size(0)
        mask_p[:, pt : pt + x.size(0)].fill_diagonal_(False)

        # logit
        # x_mag = F.softplus(x[:, :1], beta=1)
        # f_mag = F.softplus(self.feat_bank[:, :1], beta=1)
        # x_dir = F.normalize(x[:, 1:], dim=1)
        # f_dir = F.normalize(self.feat_bank[:, 1:], dim=1)
        # logits = x_mag.mm(f_mag.t()) * (x_dir.mm(f_dir.t()) - self.b_cos)

        x_norm = F.normalize(x, dim=1)
        f_norm = F.normalize(self.feat_bank, dim=1)
        logits = x_norm.mm(f_norm.t()) - self.b_cos

        logits_p = torch.masked_select(logits, mask_p)
        logits_p = (logits_p - self.m + self.b_logit) / self.r
        logits_n = torch.masked_select(logits, mask_n)
        logits_n = (logits_n + self.m + self.b_logit) * self.r

        # loss
        loss_p = F.binary_cross_entropy_with_logits(
            logits_p,
            torch.ones_like(logits_p),
        )
        loss_n = F.binary_cross_entropy_with_logits(
            logits_n,
            torch.zeros_like(logits_n),
        )
        loss = self.alpha * loss_p + (1.0 - self.alpha) * loss_n

        return self.lw * loss


if __name__ == "__main__":
    pass
