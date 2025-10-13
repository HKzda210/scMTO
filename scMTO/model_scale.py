from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scMTO.layers import ZINBLoss, TOPICLoss, MeanAct, DispAct
from torch.optim import Adam
from scMTO.preprocess import log1pnorm, log1pnormscale
from sklearn.cluster import KMeans
from scMTO.topic_function import graph_Laplacian_sparse
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

class scGSTMSCAL(nn.Module):
    def __init__(self, latent_dim, lambda1, iteration):
        super(scGSTMSCAL, self).__init__()
        self.latent_dim = latent_dim
        self.lambda1 = lambda1
        self.iteration = iteration

    def forward(self, input, Laplacian, S, Degree, n_z):
        matrix_gene_trans, matrix_cell_trans = self.initmatrices(input, n_z)
        step = 1
        loss_last = 1e9
        rec_loss = (np.linalg.norm(input - matrix_gene_trans @ matrix_cell_trans, ord='fro')**2
                     + self.lambda1 * np.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T))
        while step < self.iteration and np.abs(loss_last - rec_loss) / loss_last > 1e-3:
            loss_last = rec_loss
            matrix_gene_trans = np.multiply(matrix_gene_trans, ((input @ matrix_cell_trans.T) /
                                         (matrix_gene_trans @ matrix_cell_trans @ matrix_cell_trans.T + 1e-9)))
            matrix_cell_trans = np.multiply(matrix_cell_trans, (matrix_gene_trans.T @ input + self.lambda1 * matrix_cell_trans @ S) /
                           (matrix_gene_trans.T @ matrix_gene_trans @ matrix_cell_trans + self.lambda1 * matrix_cell_trans @ Degree + 0.1 + 1e-9))
            matrix_gene_last = matrix_gene_trans
            matrix_gene_trans = matrix_gene_trans / np.linalg.norm(matrix_gene_last, axis=0)[None, :]
            matrix_cell_trans = np.multiply(matrix_cell_trans, np.linalg.norm(matrix_gene_last, axis=0)[:, None])
            rec_loss = (np.linalg.norm(input - matrix_gene_trans @ matrix_cell_trans, ord='fro')**2
                         + self.lambda1 * np.trace(matrix_cell_trans @ Laplacian @ matrix_cell_trans.T))
            step = step + 1
        return matrix_gene_trans, matrix_cell_trans

    def initmatrices(self, input, n_z):
        u, s, V = np.linalg.svd(input, full_matrices=False)
        U = u[:, :n_z]
        V = V[:n_z, :]
        S = s[:n_z]
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
                u = x_p / x_p_nrm
                v = y_p / y_p_nrm
                sigma = m_p
            else:
                u = x_n / x_n_nrm
                v = y_n / y_n_nrm
                sigma = m_n
            lbd = sqrt(S[j] * sigma)
            matrix_gene_trans[:, j] = lbd * u
            matrix_cell_trans[j, :] = lbd * v
        avg = np.mean(input)
        matrix_gene_trans[matrix_gene_trans < 1e-8] = avg
        matrix_cell_trans[matrix_cell_trans < 1e-8] = avg
        return matrix_gene_trans, matrix_cell_trans

class AutoEncoderSCAL(nn.Module):
    def __init__(self, n_z, device, W, H):
        super(AutoEncoderSCAL, self).__init__()
        self.device = device
        self.n_z = n_z
        self.gamma = 0.1
        self.W = nn.Parameter(W)
        self.H = nn.Parameter(H)
        self.transformation_func = lambda x: F.relu(x)

        encoder1 = [256]
        self.encoder1 = nn.Sequential(nn.Linear(1000, encoder1[0]), nn.ReLU()).to(self.device)
        self.encoder1.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder1)):
            self.encoder1.extend([nn.Linear(encoder1[i - 1], encoder1[i]), nn.ReLU()])
        self.z_layer1 = nn.Linear(encoder1[-1], n_z).to(self.device)

        encoder2 = [128]
        self.encoder2 = nn.Sequential(nn.Linear(1000, encoder2[0]), nn.ReLU(inplace=True)).to(self.device)
        self.encoder2.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder2)):
            self.encoder2.extend([nn.Linear(encoder2[i-1], encoder2[i]), nn.ReLU(inplace=True)])
        self.z_layer2 = nn.Linear(encoder2[-1], n_z).to(self.device)

        encoder3 = [256]
        self.encoder3 = nn.Sequential(nn.Linear(500, encoder3[0]), nn.ReLU(inplace=True)).to(self.device)
        self.encoder3.append(nn.Dropout(p=0.2))
        for i in range(1, len(encoder3)):
            self.encoder3.extend([nn.Linear(encoder3[i-1], encoder3[i]), nn.ReLU(inplace=True)])
        self.z_layer3 = nn.Linear(encoder3[-1], n_z).to(self.device)

        decoder1 = [256]
        self.ZINB_decoder = nn.Sequential(nn.Linear(n_z, decoder1[0]), nn.ReLU()).to(self.device)
        for i in range(1, len(decoder1)):
            self.ZINB_decoder.extend([nn.Linear(decoder1[i - 1], decoder1[i]), nn.ReLU()])
        self._mean = nn.Sequential(nn.Linear(decoder1[-1], 1000), MeanAct())
        self._disp = nn.Sequential(nn.Linear(decoder1[-1], 1000), DispAct())
        self._pi = nn.Sequential(nn.Linear(decoder1[-1], 1000), nn.Sigmoid())

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

    def forward(self, x, x_raw, L, size_factor, upidx, downidx):
        X = log1pnorm(x_raw)
        Z = self.z_layer_noise(x, X)
        V = self.transformation_func(Z)
        h = self.ZINB_decoder(V + 0.1* F.relu(self.W[upidx:downidx]))
        _mea, _dis, _p = self._mean(h), self._disp(h), self._pi(h)
        ZINB_loss = self.zinbloss(x=x[0], mean=_mea, disp=_dis, pi=_p, scale_factor=size_factor, largescale=True)

        Z = self.z_layer(x, X)
        Prior = torch.randn(x[0].size()[0], self.n_z).to(self.device)
        mmd_loss = self.imq_kernel(Z, Prior, self.calculate_C(Z, Prior)) / x[0].size()[0]
        X_rec = torch.matmul(V, F.relu(self.H))
        L_coo = L.tocoo()
        row = torch.tensor(L_coo.row, dtype=torch.long)
        col = torch.tensor(L_coo.col, dtype=torch.long)
        data = torch.tensor(L_coo.data, dtype=torch.float32)
        indices = torch.stack([row, col], dim=0)
        L_torch_sparse = torch.sparse_coo_tensor(indices, data, size=L_coo.shape).to(self.device)
        OT_loss = self.topicloss(x=X, x_rec=X_rec, V=V, L=L_torch_sparse) + 1* mmd_loss

        return ZINB_loss, OT_loss, Z

class scMTOSCAL(nn.Module):
    def __init__(self, n_clusters, device, x_raw, update_step=1, W=None, H=None, L=None):
        super(scMTOSCAL, self).__init__()
        self.device = device
        self.n_z = 20
        self.scgstm = scGSTMSCAL(self.n_z, lambda1=1.0, iteration=500)
        if W is not None and H is not None and L is not None:
            self.W, self.H, self.L = W, H, L
        else: self.W, self.H = self.InitWithGSTM(x_raw)
        self.autoencoder = AutoEncoderSCAL(self.n_z, self.device, self.W, self.H)
        self.gamma_1, self.gamma_2, self.gamma_3 = 0.99, 0.01, 1.5
        self.n_clusters = n_clusters
        self.update_step = update_step
        if n_clusters is not None:
            self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.n_z))
            torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def InitWithGSTM(self, x_raw):
        raw_norm = log1pnormscale(x_raw, 1024)
        self.L, self.S, self.D = graph_Laplacian_sparse(raw_norm, n_k=10)
        H, W = self.scgstm(raw_norm.T, self.L, self.S, self.D, 20)
        H = (torch.Tensor(H)).T.to(self.device)
        W = (torch.Tensor(W)).T.to(self.device)
        return W, H

    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def cross_entropy_loss(self, p, u):
        loss = -torch.sum(p * torch.log(u), dim=1).mean()
        return loss

    def forward(self, x, raw, L, size_factor, upidx, downidx):
        self.v = 1.0
        zinb_loss, rec_loss, z = self.autoencoder(x, raw, L, size_factor, upidx, downidx)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return zinb_loss, rec_loss, q, z

    def pretrain(self, x, x2, raw_data, size_factor, pre_lr, pre_epoch, batch_size=256):
        torch.cuda.empty_cache()
        num_batches = int(np.ceil(1.0*x.shape[0]/batch_size))
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=pre_lr)
        self.embeddings = []
        print('Pre_training...')
        for epoch in range(pre_epoch):
            for batch_idx in range(num_batches):
                x1batch = x[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                x2batch =  x2[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                x_rawbatch = raw_data[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                size_factorbatch = size_factor[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                Lbatch = self.L[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0]),
                         batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                
                x1batch = torch.Tensor(x1batch).to(self.device)
                x2batch = torch.Tensor(x2batch).to(self.device)
                xbatch = [x1batch, x2batch]
                x_rawbatch = torch.Tensor(x_rawbatch).to(self.device)
                size_factorbatch = torch.Tensor(size_factorbatch).to(self.device)
                ZINB_loss, OT_loss, zbatch = self.autoencoder(xbatch, x_rawbatch, Lbatch, size_factorbatch,  
                                                                            batch_idx*batch_size, min((batch_idx+1)*batch_size, x.shape[0]))
                loss = self.gamma_1 * ZINB_loss + self.gamma_2 * OT_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == pre_epoch - 1:
                    self.embeddings.append(zbatch)
                if epoch % self.update_step == 0:
                    print(f'{epoch}: Pretrain Loss:{loss.item():.6f}')
        self.embeddings = torch.cat(self.embeddings, dim=0)
        print('Pre_train finished!')
        return self.embeddings

    def fit(self, x, x2, y, raw_data, size_factor, lr, train_epoch, batch_size=256):
        torch.cuda.empty_cache()
        print('n_clusters: ', self.n_clusters)
        if self.n_clusters is not None:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=30, random_state=666)
            y_pred = kmeans.fit_predict(self.embeddings.data.cpu().numpy())
            self.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
            self.y_pred_last = y_pred

        if y is not None:
            acc = cluster_accuracy(y, y_pred)
            nmi = normalized_mutual_info_score(y, y_pred)
            ari = adjusted_rand_score(y, y_pred)
            print(f'k-means: acc: {acc}, nmi: {nmi}, ari:{ari}')

        num_batchs = int(np.ceil(1.0*x.shape[0]/batch_size))
        optimizer = Adam(self.parameters(), lr=lr, amsgrad=True)
        embeddings = []
        print('Clustering...')
        for epoch in range(train_epoch):
            preds = []
            for batch_idx in range(num_batchs):
                x1batch =  x[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                x2batch =  x2[batch_idx*batch_size:min((batch_idx+1)*batch_size, x.shape[0])]
                x_rawbatch = raw_data[batch_idx * batch_size:min((batch_idx + 1) * batch_size, x.shape[0])]
                size_factorbatch = size_factor[batch_idx * batch_size:min((batch_idx + 1) * batch_size, x.shape[0])]
                Lbatch = self.L[batch_idx * batch_size:min((batch_idx + 1) * batch_size, x.shape[0]),
                         batch_idx * batch_size:min((batch_idx + 1) * batch_size, x.shape[0])]
                if y is not None:
                    ybatch = y[batch_idx * batch_size:min((batch_idx + 1) * batch_size, x.shape[0])]

                x1batch = torch.Tensor(x1batch).to(self.device)
                x2batch = torch.Tensor(x2batch).to(self.device)
                xbatch = [x1batch, x2batch]
                x_rawbatch = torch.Tensor(x_rawbatch).to(self.device)
                size_factorbatch = torch.Tensor(size_factorbatch).to(self.device)
                ZINB_loss, OT_loss, qbatch, zbatch = self.forward(xbatch, x_rawbatch, Lbatch, size_factorbatch, 
                                                                                batch_idx*batch_size, min((batch_idx+1)*batch_size, x.shape[0]))
                qbatch = qbatch.data
                pbatch = self.target_distribution(qbatch)
                y_pred = qbatch.cpu().detach().numpy().argmax(1)
                if y is not None:
                    acc = cluster_accuracy(ybatch, y_pred)
                    nmi = normalized_mutual_info_score(ybatch, y_pred)
                    ari = adjusted_rand_score(ybatch, y_pred)
                ZINB_loss, OT_loss, qbatch, zbatch = self.forward(xbatch, x_rawbatch, Lbatch, size_factorbatch,
                                                                                batch_idx*batch_size, min((batch_idx+1)*batch_size, x.shape[0]))
                loss_cluster = F.kl_div(qbatch.log(), pbatch, reduction='batchmean')
                loss = self.gamma_1 * ZINB_loss + self.gamma_2 * OT_loss + self.gamma_3 * loss_cluster
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                preds.append(y_pred)
                if epoch == train_epoch - 1:
                    embeddings.append(zbatch)
                    
                if epoch % self.update_step == 0:
                    print(f'{epoch}: Total Loss {loss:.4f}, '
                          f'ZINB Loss {ZINB_loss:.4f}, OT Loss {OT_loss:.4f}, '
                          f'KL Loss {loss_cluster:.4f}', end='\n')
                    if y is not None:
                        print(epoch, f':acc {acc:.10f}, nmi {nmi:.10f}, ari {ari:.10f}')
                    print()
            preds = np.concatenate(preds, axis=0)
            if epoch == 0:
                self.pred_last = preds
            else: # check stop criterion
                delta_label = np.sum(preds != self.y_pred_last).astype(np.float32) / preds.shape[0]
                if delta_label < 0.005:
                    print('delta_label ', delta_label, '< tol ', 0.005)
                    print('Reached tolerance threshold. Stopping training.')
                    break
        embeddings = torch.cat(embeddings, dim=0)
        final_acc, final_nmi, final_ari = 0, 0, 0
        if y is not None:
            final_acc = cluster_accuracy(y, preds)
            final_nmi = normalized_mutual_info_score(y, preds)
            final_ari = adjusted_rand_score(y, preds)

        return embeddings, preds, final_acc, final_nmi, final_ari









  