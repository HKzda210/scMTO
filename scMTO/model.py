from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from .layers import ZINBLoss, TOPICLoss, MeanAct, DispAct
from torch.optim import Adam
from .preprocess import log1pnorm
from sklearn.cluster import KMeans
from .topic_function import graph_Laplacian
from sklearn.utils.extmath import squared_norm
from math import sqrt

def cluster_accuracy(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array((ind[0], ind[1])).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


class scGSTM(nn.Module):
    def __init__(self, latent_dim, lambda1, iteration, device):
        super(scGSTM, self).__init__()
        self.latent_dim = latent_dim
        self.lambda1 = lambda1
        self.iteration = iteration
        self.device = device

    def init_matrices(self, input, n_z):
        u, s, Vh = torch.linalg.svd(input, full_matrices=False)
        U, V, S= u[:, :n_z], Vh[:n_z, :], s[:n_z]
        matrix_gene_trans = torch.zeros_like(U)
        matrix_cell_trans = torch.zeros_like(V)
        matrix_gene_trans[:, 0] = torch.sqrt(S[0]) * U[:, 0].abs()
        matrix_cell_trans[0, :] = torch.sqrt(S[0]) * V[0, :].abs()
        for j in range(1, n_z):
            x, y = U[:, j], V[j, :]
            x_p, y_p = torch.clamp(x, min=0), torch.clamp(y, min=0)  
            x_n, y_n = torch.clamp(x, max=0).abs(), torch.clamp(y, max=0).abs() 
            x_p_nrm, y_p_nrm = torch.sqrt(torch.sum(x_p ** 2)), torch.sqrt(torch.sum(y_p ** 2))
            x_n_nrm, y_n_nrm = torch.sqrt(torch.sum(x_n ** 2)), torch.sqrt(torch.sum(y_n ** 2))
            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
            if m_p > m_n:
                u, v, sigma = x_p / (x_p_nrm + 1e-9), y_p / (y_p_nrm + 1e-9), m_p
            else:
                u, v, sigma = x_n / (x_n_nrm + 1e-9), y_n / (y_n_nrm + 1e-9), m_n
            lbd = sqrt(S[j] * sigma)
            matrix_gene_trans[:, j] = lbd * u
            matrix_cell_trans[j, :] = lbd * v
        avg = input.mean()
        matrix_gene_trans[matrix_gene_trans < 1e-8] = avg
        matrix_cell_trans[matrix_cell_trans < 1e-8] = avg
        return matrix_gene_trans, matrix_cell_trans

    def train(self, input, Laplacian, S, Degree, n_z):
        input = torch.Tensor(input).to(self.device)
        Laplacian = torch.Tensor(Laplacian).to(self.device)

        matrix_gene_trans, matrix_cell_trans = self.init_matrices(input, n_z)
        matrix_gene_trans = torch.Tensor(matrix_gene_trans).to(self.device)
        matrix_cell_trans = torch.Tensor(matrix_cell_trans).to(self.device)
        step = 1
        loss_last = 1e9

        rec_loss = (
            torch.norm(input - matrix_gene_trans @ matrix_cell_trans, p='fro') ** 2
            + self.lambda1 * torch.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T)
        )

        while step < self.iteration and torch.abs(loss_last - rec_loss) / loss_last > 1e-3: 
            loss_last = rec_loss
            # update W and H
            matrix_cell_trans, matrix_gene_trans = self.forward(input, matrix_cell_trans, matrix_gene_trans, S, Degree)
            # Update reconstruction loss
            rec_loss = (
                torch.norm(input - matrix_gene_trans @ matrix_cell_trans, p='fro') ** 2
                + self.lambda1 * torch.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T)
            )
            step += 1

        return matrix_gene_trans.cpu().detach().numpy(), matrix_cell_trans.cpu().detach().numpy()
    
    def forward(self, input, matrix_cell_trans, matrix_gene_trans, S, Degree):
        # Update matrix_gene_trans
        numerator_gene = input @ matrix_cell_trans.T
        denominator_gene = matrix_gene_trans @ matrix_cell_trans @ matrix_cell_trans.T + 1e-9
        matrix_gene_trans_new = matrix_gene_trans * (numerator_gene / denominator_gene)

        # Update matrix_cell_trans
        numerator_cell = matrix_gene_trans.T @ input + self.lambda1 * matrix_cell_trans @ S
        denominator_cell = (
            matrix_gene_trans.T @ matrix_gene_trans @ matrix_cell_trans
            + self.lambda1 * matrix_cell_trans @ Degree
            + 0.1 + 1e-9
        )
        matrix_cell_trans_new = matrix_cell_trans * (numerator_cell / denominator_cell)
        
        matrix_gene_last = matrix_gene_trans_new
        norm_gene = torch.norm(matrix_gene_last, dim=0, keepdim=True)
        matrix_gene_trans_new /= norm_gene
        matrix_cell_trans_new *= norm_gene.T
        
        return matrix_cell_trans_new, matrix_gene_trans_new

    def npinit_matrices(self, input, n_z):
        u, s, V = np.linalg.svd(input, full_matrices=False)
        U, V, S = u[:, :n_z], V[:n_z, :], s[:n_z]
        matrix_gene_trans = np.zeros_like(U)
        matrix_cell_trans = np.zeros_like(V)
        matrix_gene_trans[:, 0] = sqrt(S[0]) * np.abs(U[:, 0])
        matrix_cell_trans[0, :] = sqrt(S[0]) * np.abs(V[0, :])
        for j in range(1, n_z):
            x, y = U[:, j], V[j, :]
            x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
            x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
            x_p_nrm, y_p_nrm = sqrt(squared_norm(x_p)), sqrt(squared_norm(y_p))
            x_n_nrm, y_n_nrm = sqrt(squared_norm(x_n)), sqrt(squared_norm(y_n))
            m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
            if m_p > m_n:
                u, v, sigma = x_p / x_p_nrm, y_p / y_p_nrm, m_p
            else:
                u, v, sigma = x_n / x_n_nrm, y_n / y_n_nrm, m_n
            lbd = sqrt(S[j] * sigma)
            matrix_gene_trans[:, j] = lbd * u
            matrix_cell_trans[j, :] = lbd * v
        avg = np.mean(input)
        matrix_gene_trans[matrix_gene_trans < 1e-8] = avg
        matrix_cell_trans[matrix_cell_trans < 1e-8] = avg
        return matrix_gene_trans, matrix_cell_trans

    def nptrain(self, input, Laplacian, S, Degree, n_z):
        # Update cell topic matrix and topic gene matrix
        matrix_gene_trans, matrix_cell_trans = self.npinit_matrices(input, n_z)
        step = 1
        loss_last = 1e9
        rec_loss = (np.linalg.norm(input - matrix_gene_trans @ matrix_cell_trans, ord='fro')**2
                     + self.lambda1 * np.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T))

        while step < self.iteration and np.abs(loss_last - rec_loss) / loss_last > 1e-3:
            loss_last = rec_loss
            matrix_gene_trans = matrix_gene_trans * ((input @ matrix_cell_trans.T) /
                                         (matrix_gene_trans @ matrix_cell_trans @ matrix_cell_trans.T + 1e-9))
            matrix_cell_trans = (matrix_cell_trans * (matrix_gene_trans.T @ input + self.lambda1 * matrix_cell_trans @ S) /
                           (matrix_gene_trans.T @ matrix_gene_trans @ matrix_cell_trans + self.lambda1 * matrix_cell_trans @ Degree + 0.1 + 1e-9))

            matrix_gene_last = matrix_gene_trans
            matrix_gene_trans = matrix_gene_trans / np.linalg.norm(matrix_gene_last, axis=0)[None, :]
            matrix_cell_trans = matrix_cell_trans * np.linalg.norm(matrix_gene_last, axis=0)[:, None]

            rec_loss = (np.linalg.norm(input - matrix_gene_trans @ matrix_cell_trans, ord='fro')**2
                         + self.lambda1 * np.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T))
            step = step + 1

        return matrix_gene_trans, matrix_cell_trans

    def npforward(self, input, matrix_cell_trans, matrix_gene_trans, S, Degree):
        S = S.cpu().detach().numpy()
        Degree = Degree.cpu().detach().numpy()
        matrix_gene_trans = matrix_gene_trans * ((input @ matrix_cell_trans.T) /
                                        (matrix_gene_trans @ matrix_cell_trans @ matrix_cell_trans.T + 1e-9))
        matrix_cell_trans = (matrix_cell_trans * (matrix_gene_trans.T @ input + self.lambda1 * matrix_cell_trans @ S) /
                        (matrix_gene_trans.T @ matrix_gene_trans @ matrix_cell_trans + self.lambda1 * matrix_cell_trans @ Degree + 0.1 + 1e-9))
        
        matrix_gene_last = matrix_gene_trans
        matrix_gene_trans = matrix_gene_trans / np.linalg.norm(matrix_gene_last, axis=0)[None, :]
        matrix_cell_trans = matrix_cell_trans * np.linalg.norm(matrix_gene_last, axis=0)[:, None]
        return matrix_cell_trans, matrix_gene_trans

class AutoEncoder(nn.Module):
    def __init__(self, n_z, cell_topic_embedding, topic_gene_embedding, device):
        super(AutoEncoder, self).__init__()
        self.n_z = n_z
        self.device = device
        # self.cell_topic_embedding = cell_topic_embedding
        self.cell_topic_embedding = nn.Parameter(cell_topic_embedding)
        self.topic_gene_embedding = nn.Parameter(topic_gene_embedding)
        self.lambda_v, self.lambda_w, self.lambda_z = 1.0, 0.1, 0.9
        self.transformation_func = lambda x: F.relu(x)

        encoder1 = [512]
        self.encoder1 = nn.Sequential(nn.Linear(2000, encoder1[0]), nn.ReLU(inplace=True)).to(self.device)
        self.encoder1.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder1)):
            self.encoder1.extend([nn.Linear(encoder1[i-1], encoder1[i]), nn.ReLU(inplace=True)])
        self.z_layer1 = nn.Linear(encoder1[-1], self.n_z).to(self.device)

        encoder2 = [128]
        self.encoder2 = nn.Sequential(nn.Linear(2000, encoder2[0]), nn.ReLU(inplace=True)).to(self.device)
        self.encoder2.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder2)):
            self.encoder2.extend([nn.Linear(encoder2[i-1], encoder2[i]), nn.ReLU(inplace=True)])
        self.z_layer2 = nn.Linear(encoder2[-1], self.n_z).to(self.device)

        encoder3 = [256]
        self.encoder3 = nn.Sequential(nn.Linear(500, encoder3[0]), nn.ReLU(inplace=True)).to(self.device)
        self.encoder3.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder3)):
            self.encoder3.extend([nn.Linear(encoder3[i-1], encoder3[i]), nn.ReLU(inplace=True)])
        self.z_layer3 = nn.Linear(encoder3[-1], self.n_z).to(self.device)

        decoder1 = [512]
        self.ZINB_decoder = nn.Sequential(nn.Linear(self.n_z, decoder1[0]), nn.ReLU(inplace=True)).to(self.device)
        for i in range(1, len(decoder1)):
            self.ZINB_decoder.extend([nn.Linear(decoder1[i-1], decoder1[i]), nn.ReLU(inplace=True)])
        self._mean = nn.Sequential(nn.Linear(decoder1[-1], 2000), MeanAct())
        self._disp = nn.Sequential(nn.Linear(decoder1[-1], 2000), DispAct())
        self._pi = nn.Sequential(nn.Linear(decoder1[-1], 2000), nn.Sigmoid())

        self.zinbloss = ZINBLoss()
        self.topicloss = TOPICLoss()
    
    def z_layer_noise(self, x, X):
        x1, x2, x3 = x[0], X, x[1]
        Z_1 = self.z_layer1(self.encoder1(x1 + torch.randn_like(x1)))
        Z_2 = self.z_layer2(self.encoder2(x2 + torch.randn_like(x2)))
        Z_3 = self.z_layer3(self.encoder3(x3 + torch.randn_like(x3)))
        Z = 1.5* Z_1 + 1.0* Z_2 + 1.0* Z_3
        return Z
    
    def z_layer(self, x, X):
        x1, x2, x3  = x[0], X,  x[1]
        Z_1 = self.z_layer1(self.encoder1(x1))
        Z_2 = self.z_layer2(self.encoder2(x2))
        Z_3 = self.z_layer3(self.encoder3(x3))
        Z = 1.5* Z_1 + 1.0* Z_2 + 1.0* Z_3
        return Z

    def calculate_C(self, X, Y):
        dists_X = torch.sum(X**2, dim=1, keepdim=True)  
        dists_Y = torch.sum(Y**2, dim=1, keepdim=True)  
        dists_XY = torch.mm(X, Y.t()) 
        dists = dists_X + dists_Y.t() - 2 * dists_XY  
        max_distance = torch.sqrt(dists).max() 
        return max_distance.item()

    def imq_kernel(self, X, Y, C):
        batch_size = X.size(0)
        norms_x = X.pow(2).sum(1, keepdim=True)  
        prods_x = torch.mm(X, X.t()) 
        dists_x = norms_x + norms_x.t() - 2 * prods_x
        norms_y = Y.pow(2).sum(1, keepdim=True)  
        prods_y = torch.mm(Y, Y.t())  
        dists_y = norms_y + norms_y.t() - 2 * prods_y
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
        stats = 0
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)
        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1
        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2
        return stats

    def forward(self, x, x_raw, L):
        X = log1pnorm(x_raw)
        Z = self.z_layer_noise(x, X)
        V = self.transformation_func(Z)
        zinb_rec = self.ZINB_decoder(self.lambda_v* V + self.lambda_w* F.relu(self.cell_topic_embedding))
        ZINB_loss = self.zinbloss(x=x[0], mean=self._mean(zinb_rec), disp=self._disp(zinb_rec), pi=self._pi(zinb_rec))

        Z = self.z_layer(x, X)
        P = torch.randn(x[0].size()[0], self.n_z).to(self.device)
        mmdloss = self.imq_kernel(Z, P, self.calculate_C(Z, P)) / x[0].size()[0]
        topic_rec = F.relu(torch.matmul(V, F.relu(self.topic_gene_embedding)))
        OT_loss = self.topicloss(x=X, x_rec=topic_rec, V=V, L=L) + self.lambda_z* mmdloss

        return ZINB_loss, OT_loss, Z, F.relu(self.cell_topic_embedding), F.relu(self.topic_gene_embedding)

class scMTO(nn.Module):
    def __init__(self, n_z, n_clusters, x_raw, device, update_step=1):
        super(scMTO, self).__init__()
        self.n_z = n_z
        self.device = device
        self.n_clusters = n_clusters
        self.update_step = update_step
        self.scgstm = scGSTM(self.n_z, lambda1=1.0, iteration=1000, device=self.device)
        self.cell_topic_embedding, self.topic_gene_embedding = self.InitWithGSTM(x_raw)
        self.autoencoder = AutoEncoder(self.n_z, self.cell_topic_embedding, self.topic_gene_embedding, self.device)
        self.gamma_1, self.gamma_2, self.gamma_3 = 0.99, 0.01, 1.5

        if n_clusters is not None:
            self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def InitWithGSTM(self, x_raw):
        x_raw = torch.Tensor(x_raw).to(self.device)
        raw_norm = log1pnorm(x_raw).cpu().detach().numpy()
        print('Calculating the cell graph...')
        self.L, self.S, self.D = graph_Laplacian(X=raw_norm, n_k=10)
        print('Topic modeling...')
        H, W = self.scgstm.nptrain(raw_norm.T, self.L, self.S, self.D, self.n_z)
        self.L = torch.Tensor(self.L).to(self.device)
        topic_gene_embedding = (torch.Tensor(H)).T.to(self.device)
        cell_topic_embedding = (torch.Tensor(W)).T.to(self.device)
        return cell_topic_embedding, topic_gene_embedding
        
    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def forward(self, x, raw):
        zinb_loss, ot_loss, z, cell_topic_w, topic_gene_h = self.autoencoder(x, raw, self.L)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1.0)
        q = q.pow((1.0 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return zinb_loss, ot_loss, q, z, cell_topic_w, topic_gene_h

    def pretrain(self, x, raw_data, pre_lr, pre_epoch):
        x = [torch.Tensor(x[0]).to(self.device), torch.Tensor(x[1]).to(self.device)]
        x_raw = torch.Tensor(raw_data).to(self.device)

        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=pre_lr)
        print('scMTO pretraining...')
        for epoch in range(pre_epoch):
            torch.cuda.empty_cache() 
            zinb_loss, ot_loss, _, _, _ = self.autoencoder(x, x_raw, self.L)
            loss = self.gamma_1* zinb_loss + self.gamma_2* ot_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % self.update_step == 0:
                print(f'{epoch}: Pretraining Loss:{loss.item():.6f}')
        print('Pretrain finished!\n')

    def fit(self, x, y, raw_data, lr, train_epoch, tol=0.005):
        data = [torch.Tensor(x[0]).to(self.device), torch.Tensor(x[1]).to(self.device)]
        raw = torch.Tensor(raw_data).to(self.device)

        with torch.no_grad():
            zinb_loss, ot_loss, z, _, _ = self.autoencoder(data, raw, self.L)

        if self.n_clusters is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=30, random_state=666)
            y_pred = kmeans.fit_predict(z.data.cpu().numpy())
            self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
            self.y_pred_last = y_pred

        if y is not None:
            acc = cluster_accuracy(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_rand_score(y, y_pred)
            print(f'Clustering initialization: K-means: ACC: {acc}, NMI: {nmi}, ARI:{ari}')

        optimizer = Adam(self.parameters(), lr=lr, amsgrad=True)
        print('scMTO clustering...')
        for epoch in range(train_epoch):
            zinb_loss, ot_loss, q, z, cell_topic_w, topic_gene_h = self.forward(data, raw)
            p = self.target_distribution(q.data)
            y_pred = q.data.cpu().detach().numpy().argmax(1)

            # check stop criterion
            delta_label = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            if epoch > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
            self.pred_last = y_pred
            if y is not None:
                acc = cluster_accuracy(y, y_pred)
                nmi = normalized_mutual_info_score(y, y_pred)
                ari = adjusted_rand_score(y, y_pred)

            zinb_loss, ot_loss, q, z, cell_topic_w, topic_gene_h = self.forward(data, raw)
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

            loss = self.gamma_1* zinb_loss + self.gamma_2* ot_loss + self.gamma_3* kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % self.update_step == 0:
                print(epoch, f':ACC {acc:.10f}, NMI {nmi:.10f}, ARI {ari:.10f}')
                print(f'{epoch}: Total Loss {loss:.4f}, '
                      f'ZINB Loss {zinb_loss:.4f}, OT Loss {ot_loss:.4f}, KL Loss {kl_loss:.4f}', end='\n')

        final_acc, final_nmi, final_ari = 0, 0, 0
        if y is not None:
            final_acc = cluster_accuracy(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
            final_ari = adjusted_rand_score(y, y_pred)

        return z, cell_topic_w, topic_gene_h, self.pred_last, final_acc, final_nmi, final_ari
    


