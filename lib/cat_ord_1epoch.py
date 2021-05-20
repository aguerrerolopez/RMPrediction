#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:26:30 2020

@author: Alejandro Guerrero-López
"""

# -- coding: utf-8 --
import torch
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import copy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder


class NeuralNetworks(torch.nn.Module):
    def __init__(self, xga_dim, tgx_dim, in_dim, out_dim, mean=False, cov=False,
                 tgx=False, cat=True, ordi=False, logger=None):
        super().__init__()

        self.train_mean = mean
        self.train_cov = cov
        self.train_tgx = tgx
        # Common layers for both mu and sigma.
        if self.train_mean or self.train_cov:
            self.hidden_1 = torch.nn.Linear(in_dim, xga_dim[0])
            # self.hidden_2 = torch.nn.Linear(xga_dim[0], xga_dim[1])
            # self.hidden_3 = torch.nn.Linear(xga_dim[1], xga_dim[2])
            self.output = torch.nn.Linear(xga_dim[-1], out_dim)
            # self.output = torch.nn.Linear(in_dim, out_dim)
        # Output activation for sigma.
        if self.train_cov:
            self.out_act = torch.nn.Softplus()
        # Layers for t.
        if self.train_tgx:
            # self.hidden_1 = torch.nn.Linear(in_dim, tgx_dim[0])
            # self.hidden_2 = torch.nn.Linear(tgx_dim[0], tgx_dim[1])
            # self.hidden_3 = torch.nn.Linear(tgx_dim[1], tgx_dim[2])
            # self.output = torch.nn.Linear(tgx_dim[-1], out_dim)
            self.output = torch.nn.Linear(in_dim, out_dim)
            if cat:
                self.out_act = torch.nn.LogSoftmax(dim=1)
            if ordi:
                self.out_act = torch.nn.Softplus()

        # Common activation after each hidden layer
        self.activation = torch.nn.ReLU()
        self.activation2 = torch.nn.Tanh()
        self.logger = logger
        self.mu, self.sigma, self.tgx = None, None, None

    def forward(self, input_vector, logger=None):
        if torch.isnan(input_vector).any():
            self.logger.error("ERROR IN FORWARD: the input vector has nans")
            self.logger.info(input_vector)
        if self.train_mean:
            # Layer 1
            self.mu = self.hidden_1(input_vector)
            if torch.isnan(self.mu).any():
                self.logger.error("ERROR IN FORWARD: apply the hidden1 layer return nans")
                self.logger.info(self.mu)
                self.logger.info("input vector")
                self.logger.info(input_vector)
                self.logger.info("weights")
                self.logger.info(self.hidden_1.weight)
                self.logger.info("bias")
                self.logger.info(self.hidden_1.bias)
            self.mu = self.activation(self.mu)
            if torch.isnan(self.mu).any():
                self.logger.error("ERROR IN FORWARD: the activation layer return nans")
                self.logger.info(self.mu)
            # self.mu = self.activation(self.hidden_1(input_vector))
            # Layer 2
            # self.mu = self.activation(self.hidden_2(self.mu))
            # # Layer 3
            # self.mu = self.activation(self.hidden_3(self.mu))
            # Output layer
            # self.mu = self.dropout(self.mu)
            self.mu = self.output(self.mu)
            if torch.isnan(self.mu).any():
                self.logger.error("ERROR IN FORWARD: the output mu layer has nans")
                self.logger.info(self.mu)
            # self.mu = self.output(input_vector)
        if self.train_cov:
            # Layer 1
            self.sigma = self.activation(self.hidden_1(input_vector))
            if torch.isnan(self.sigma).any():
                self.logger.error("ERROR IN FORWARD: the first sigma layer has nans")
                self.logger.info(self.sigma)
            # # Layer 2
            # self.sigma = self.activation(self.hidden_2(self.sigma))
            # # Layer 3
            # self.sigma = self.activation2(self.hidden_3(self.sigma))
            # Output layer
            # self.sigma = self.dropout(self.sigma)
            self.sigma = self.out_act(self.output(self.sigma))
            # self.sigma = self.out_act(self.output(input_vector))
        if self.train_tgx:
            # Layer 1
            # self.tgx = self.activation(self.hidden_1(input_vector))
            # Layer 2
            # self.tgx = self.activation(self.hidden_2(self.tgx))
            # # Layer 3
            # self.tgx = self.activation(self.hidden_3(self.tgx))
            # Output layer
            # self.tgx = self.dropout(self.tgx)
            # self.tgx = self.out_act(self.output(self.tgx))
            self.tgx = self.out_act(self.output(input_vector))


class CategoricalNN:
    def __init__(self, d_x, xga, tgx, lr, t_raw=None, categories="auto", store=True, SS=False, mask=None, cat=False,
                 ordi=False, logger=None):
        self.logger = logger
        # Categorical or ordinal module:
        self.cat, self.ord = cat, ordi
        # Looks for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Categorical module using: ", self.device)
        # Whether to store or not the loss changes
        self.store = store
        # We compute the one-hot encoder of this t_raw Nx1. Now we have NxC.
        if self.cat:
            self.binarizer = OneHotEncoder(sparse=False, categories="auto")
            self.binarizer.fit(t_raw.reshape(-1, 1))
            self.t_ohe = torch.from_numpy(self.binarizer.transform(t_raw.reshape(-1, 1))).double()
        if self.ord:
            self.sigmoid = torch.nn.Sigmoid()
            c = np.max(t_raw+1)
            self.t_ohe = torch.from_numpy((np.arange(c) < np.array(t_raw+1).reshape(-1, 1)).astype(int)).double()
            self.lb = OneHotEncoder(sparse=False, categories="auto")
        self.t_ohe.to(self.device)
        # Take care if its semisupervised or not
        self.SS = SS
        if self.SS:
            # Mask to know which points are missing
            self.mask = mask
            # Create a numpy array to store the X that we are going to return to SSHIBA: mean and cov.
            self.X_aprox = [np.zeros((t_raw.shape[0], int(d_x))), np.zeros((t_raw.shape[0], int(d_x)))]
        # Define in and out dimensions of each NN
        dim_x = [int(d_x) + self.t_ohe.shape[1], int(d_x)]
        dim_t = [int(d_x), self.t_ohe.shape[1]]
        # Instance the three NNs
        self.mu_nn = NeuralNetworks(xga_dim=xga, tgx_dim=tgx, in_dim=dim_x[0], out_dim=dim_x[1], mean=True, cat=self.cat, ordi=self.ord, logger=self.logger)
        self.sigma_nn = NeuralNetworks(xga_dim=xga, tgx_dim=tgx, in_dim=dim_x[0], out_dim=dim_x[1], cov=True, cat=self.cat, ordi=self.ord, logger=self.logger)
        self.tgx_nn = NeuralNetworks(xga_dim=xga, tgx_dim=tgx, in_dim=dim_t[0], out_dim=dim_t[1], tgx=True, cat=self.cat, ordi=self.ord, logger=self.logger)
        # Instance the optimizers of the NNs
        self.opt_mean = torch.optim.Adam(self.mu_nn.parameters(), lr[0])
        self.opt_cov = torch.optim.Adam(self.sigma_nn.parameters(), lr[1])
        self.opt_tgx = torch.optim.Adam(self.tgx_nn.parameters(), lr[2])
        # Move the NNs to GPU if exists
        self.tgx_nn.to(self.device)
        self.mu_nn.to(self.device)
        self.sigma_nn.to(self.device)
        # Lists to store the loss
        # TODO: cambiar esto para que se escoja cuando se llama a la función
        self.store = False
        if self.store:
            self.full_loss = []
            self.tgx_loss = []
            self.xga_loss = []
            self.entropy_loss = []
        else:
            self.full_loss = 0

    def mc_1ord(self, targets, iterations=10):
        aux_zeros = torch.zeros((self.mu_nn.mu.shape[0], 1)).to(self.device)
        aux_ones = torch.ones((self.mu_nn.mu.shape[0], 1)).to(self.device)

        # Get a sample of X
        x_aprox = self.mu_nn.mu + torch.empty(1).normal_().to(self.device) * torch.sqrt(self.sigma_nn.sigma)
        # Forward
        self.tgx_nn.forward(x_aprox)

        # Calcular R términos de theta_0 hasta theta_r-1 - h
        theta_terms = torch.cumsum(self.tgx_nn.tgx[:, 1:], dim=1)
        h_value = self.tgx_nn.tgx[:, 0].unsqueeze(1)

        sigmoid_terms = self.sigmoid(theta_terms - h_value)

        probs = torch.cat((sigmoid_terms, aux_ones), 1) - torch.cat((aux_zeros, sigmoid_terms), 1)
        logprobs = torch.log(probs)

        return torch.sum(logprobs[targets])

    def mc_it(self, targets, iterations=10):
        accumulative_loss = 0
        for i in range(int(iterations)):
            x_aprox = self.mu_nn.mu + torch.empty(1).normal_().to(self.device) * torch.sqrt(self.sigma_nn.sigma)
            self.tgx_nn.forward(x_aprox)
            accumulative_loss += torch.sum(self.tgx_nn.tgx[targets])
        return accumulative_loss / iterations

    def mc_1cat(self, targets):
        # Get a sample of X
        x_aprox = self.mu_nn.mu + torch.empty(1).normal_().to(self.device) * torch.sqrt(self.sigma_nn.sigma)
        # Forward
        self.tgx_nn.forward(x_aprox)
        # Return the loss term
        return torch.sum(self.tgx_nn.tgx[targets])

    def elbo(self, z, w, targets, it=10):
        # E[T | X]
        if self.cat:
            exp_tgx = self.mc_1cat(targets=targets)
        if self.ord:
            exp_tgx = self.mc_1ord(targets=targets)
        # exp_tgx = self.mc_it(targets=targets, iterations=it)
        # E[X | W, Z, tau]
        exp_xga = -1 * 0.5 * (torch.sum(self.sigma_nn.sigma) +
                              torch.sum(torch.pow(self.mu_nn.mu, 2))) + 1 * torch.sum(
                              torch.diag(self.mu_nn.mu @ w.float() @ z.float().T))
        # H(q(X))
        entropy_xga = 0.5 * torch.sum(torch.log(self.sigma_nn.sigma))
        # Construct the loss
        loss = -1 * (exp_tgx + exp_xga + entropy_xga)
        return -exp_tgx, -exp_xga, -entropy_xga, loss

    def aprox(self, logger):
        if self.SS:
            if np.any(np.isnan(self.X_aprox[0])):
                self.logger.error("ERROR IN APROX METHOD: X_aprox has nans")
                self.logger.info("X aprox mean shape")
                self.logger.info(self.X_aprox[0].shape)
                self.logger.info("X aprox cov shape")
                self.logger.info(self.X_aprox[1].shape)
                self.logger.info("X mean value")
                self.logger.info(self.X_aprox[0])
                self.logger.info("X cov value")
                self.logger.info(self.X_aprox[1])

            return self.X_aprox
        else:
            return self.mu_nn.mu.data.cpu().numpy(), self.sigma_nn.sigma.data.cpu().numpy()

    def predict_proba(self, Z, W, b=None, samples=5):
        with torch.no_grad():
            self.tgx_nn.eval()
            if b is not None:
                x_aprox = torch.from_numpy(Z @ W.T + b)
            else:
                x_aprox = torch.from_numpy(Z @ W.T)

            self.tgx_nn.forward(x_aprox.float().to(self.device))

            if self.ord:
                aux_zeros = torch.zeros((Z.shape[0], 1)).to(self.device)
                aux_ones = torch.ones((Z.shape[0], 1)).to(self.device)

                theta_terms = torch.cumsum(self.tgx_nn.tgx[:, 1:], dim=1)
                h_value = self.tgx_nn.tgx[:, 0].unsqueeze(1)

                sigmoid_terms = self.sigmoid(theta_terms - h_value)
                probs = torch.cat((sigmoid_terms, aux_ones), 1) - torch.cat((aux_zeros, sigmoid_terms), 1)
                probs = torch.clamp(probs, min=1e-6, max=1.0)

            if self.cat:
                probs = torch.exp(self.tgx_nn.tgx)

            return probs.data.cpu().numpy()

    def predict(self, X_in, numpy=False):
        with torch.no_grad():
            self.tgx_nn.eval()
            self.tgx_nn.forward(X_in.float().to(self.device))
            if self.ord:
                aux_zeros = torch.zeros((X_in.shape[0], 1)).to(self.device)
                aux_ones = torch.ones((X_in.shape[0], 1)).to(self.device)

                theta_terms = torch.cumsum(self.tgx_nn.tgx[:, 1:], dim=1)
                h_value = self.tgx_nn.tgx[:, 0].unsqueeze(1)

                sigmoid_terms = self.sigmoid(theta_terms - h_value)
                probs = torch.cat((sigmoid_terms, aux_ones), 1) - torch.cat((aux_zeros, sigmoid_terms), 1)
                probs = torch.clamp(probs, min=1e-6, max=1.0)
            if self.cat:
                probs = torch.exp(self.tgx_nn.tgx)

            return np.argmax(probs.data.cpu().numpy(), axis=1)

    def SGD_step(self, Z, W, b, t_raw, steps=500, mc_it=1, data_per=0.1, logger=None):
        # Construct the input to our NN system: Y concat T
        z = torch.from_numpy(Z["mean"]).to(self.device)
        w = torch.from_numpy(W["mean"]).to(self.device)
        # y is Z@W.t + b
        y = (z @ w.T).to(self.device) + torch.from_numpy(b).to(self.device)

        if self.SS:
            if self.ord:
                self.lb.fit(torch.sum(self.t_ohe, axis=1).view(-1,1))
            # If we are on  SS, we only pass to the NN the points that are observed
            nn_in = torch.cat((y[~self.mask].double().to(self.device), self.t_ohe[~self.mask].double().to(self.device)),
                              1).float()
            permutation = torch.randperm(self.t_ohe[~self.mask].shape[0])[
                          :int(self.t_ohe[~self.mask].shape[0] * data_per)]
            dataset = TensorDataset(nn_in[permutation], self.t_ohe[~self.mask][permutation], z[~self.mask][permutation])
            loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)
        else:
            if self.ord:
                self.lb.fit(torch.sum(self.t_ohe, axis=1).view(-1, 1))
            nn_in = torch.cat((y.double().to(self.device), self.t_ohe.double().to(self.device)), 1).float()
            permutation = torch.randperm(self.t_ohe.shape[0])[:int(self.t_ohe.shape[0] * data_per)]
            dataset = TensorDataset(nn_in[permutation], self.t_ohe[permutation], z[permutation])
            loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

        mean_model = copy.deepcopy(self.mu_nn.state_dict())
        cov_model = copy.deepcopy(self.sigma_nn.state_dict())
        tgx_model = copy.deepcopy(self.tgx_nn.state_dict())
        mean_opt = copy.deepcopy(self.opt_mean.state_dict())
        cov_opt = copy.deepcopy(self.opt_cov.state_dict())
        tgx_opt = copy.deepcopy(self.opt_tgx.state_dict())

        for e in range(int(steps)):
            # Train mode
            self.mu_nn.train()
            self.sigma_nn.train()
            self.tgx_nn.train()
            for batch_x, batch_y, batch_z in loader:
                batch_x, batch_z = batch_x.to(self.device), batch_z.to(self.device)
                if self.ord:
                    batch_y = torch.from_numpy(self.lb.transform(torch.sum(batch_y, axis=1).view(-1, 1)).astype(bool))
                if self.cat:
                    batch_y = batch_y.type(torch.bool).to(self.device)
                # Set Zero grad
                self.opt_mean.zero_grad()
                self.opt_cov.zero_grad()
                self.opt_tgx.zero_grad()
                # Pass the data forward the mu and sigma NNs
                self.mu_nn.forward(batch_x)
                self.sigma_nn.forward(batch_x)
                # Calculate the loss
                tgx_i, xga_i, entropy_i, loss = self.elbo(z=batch_z, w=w, targets=batch_y, it=mc_it)
                # Calculate gradients
                loss.backward()
                # Optimize the three NNs
                self.opt_tgx.step()
                self.opt_mean.step()
                self.opt_cov.step()

            # Store results to plot
            if self.store:
                self.tgx_loss.append(tgx_i.data.cpu().numpy())
                self.xga_loss.append(xga_i.data.cpu().numpy())
                self.entropy_loss.append(entropy_i.data.cpu().numpy())
                self.full_loss.append(-loss.data.cpu().numpy())

        # Provide q(X) to SSHIBA
        # We evaluate the
        self.mu_nn.eval()
        self.sigma_nn.eval()

        self.mu_nn.forward(nn_in)
        self.sigma_nn.forward(nn_in)
        # If it is semisupervised, update the values of the missing categories.
        if self.SS:
            # OBSERVED POINTS: the output of both NNs
            self.X_aprox[0][~self.mask] = self.mu_nn.mu.data.cpu().numpy()
            self.X_aprox[1][~self.mask] = self.sigma_nn.sigma.data.cpu().numpy()
            if np.any(np.isnan(self.X_aprox[0][~self.mask])):
                self.logger.error("ERROR IN SGD_step: THE SUPERVISED HAS PROBLEMS X_aprox[0][~self.mask] has nans")
                self.logger.info("X_aprox[0][~self.mask]  shape")
                self.logger.info(self.X_aprox[0][~self.mask].shape)
                self.logger.info("X_aprox[1][~self.mask] cov shape")
                self.logger.info(self.X_aprox[1][~self.mask].shape)
                self.logger.info("X_aprox[0][~self.mask] value")
                self.logger.info(self.X_aprox[0][~self.mask])
                self.logger.info("X_aprox[1][~self.mask] value")
                self.logger.info(self.X_aprox[1][~self.mask])
                self.logger.info("nn in value")
                self.logger.info(nn_in)
                self.logger.info("y[~self.mask] value")
                self.logger.info(y[~self.mask])
                self.logger.info("y max value")
                self.logger.info(torch.max(y[~self.mask]))
                self.logger.info("Is there any nan value at the input?")
                self.logger.info(torch.any(torch.isnan(y[~self.mask])))
                self. logger.info("t_ohe[~self.mask] value")
                self.logger.info(self.t_ohe[~self.mask])
                self.logger.info("t max value")
                self.logger.info(torch.max(self.t_ohe[~self.mask]))
                self.logger.info("Is there any nan value at the input?")
                self.logger.info(torch.any(torch.isnan(self.t_ohe[~self.mask])))

            # MISSING POINTS: mean = Z@W.T + b ; cov = tau
            # TODO: AHORA MISMO ENVIAMOS LA MISMA Y SIEMPRE, POR TANTO NO SE ACTUALIZA.
            # self.X_aprox[0][self.mask] = y[self.mask].data.cpu().numpy()
            # self.X_aprox[1][self.mask] = np.ones((self.X_aprox[1][self.mask].shape))
            # # TODO: PROBAR DE ACTUALIZAR LA Y CON LA T_INPUTTED
            self.mu_nn.eval()
            self.sigma_nn.eval()
            if self.cat:
                self.t_ohe = torch.from_numpy(self.binarizer.transform(t_raw.reshape(-1, 1))).double()
            if self.ord:
                c = np.max(np.unique(t_raw + 1))
                self.t_ohe = torch.from_numpy((np.arange(c) < np.array(t_raw + 1).reshape(-1, 1)).astype(int)).double()
            ss_in = torch.cat((y[self.mask].double().to(self.device), self.t_ohe[self.mask].double().to(self.device)),
                              1).float()
            self.mu_nn.forward(ss_in)
            self.sigma_nn.forward(ss_in)
            self.X_aprox[0][self.mask] = self.mu_nn.mu.data.cpu().numpy()
            self.X_aprox[1][self.mask] = self.sigma_nn.sigma.data.cpu().numpy()
            if np.any(np.isnan(self.X_aprox[0][~self.mask])):
                logger.error("ERROR IN SGD_step: THE SEMISUPERVISED HAS PROBLEMS X_aprox[0][self.mask] has nans")
                logger.info("X_aprox[0][self.mask]  shape")
                logger.info(self.X_aprox[0][self.mask].shape)
                logger.info("X_aprox[1][self.mask] cov shape")
                logger.info(self.X_aprox[1][self.mask].shape)
                logger.info("X_aprox[0][self.mask] value")
                logger.info(self.X_aprox[0][~self.mask])
                logger.info("X_aprox[1][self.mask] value")
                logger.info(self.X_aprox[1][~self.mask])
                logger.info("ss in value")
                logger.info(ss_in)
                logger.info("y[mask] value")
                logger.info(y[self.mask])
                logger.info("t_ohe value")
                logger.info(self.t_ohe[self.mask])

        return loss.data.cpu().numpy(), mean_model, cov_model, tgx_model, mean_opt, cov_opt, tgx_opt

        # if self.store:
        #     from matplotlib import pyplot as plt
        #     plt.figure()
        #     plt.subplot(2, 2, 1)
        #     plt.title("E[t|X]")
        #     plt.plot(tgx_loss)
        #
        #     plt.subplot(2, 2, 2)
        #     plt.title("E[X|W, Z]")
        #     plt.plot(xga_loss)
        #
        #     plt.subplot(2, 2, 3)
        #     plt.title("H(q(x))")
        #     plt.plot(entropy_loss)
        #
        #     plt.subplot(2, 2, 4)
        #     plt.title("Categorical ELBO")
        #     plt.xlabel("Iterations")
        #     plt.ylabel("LB")
        #     plt.plot(full_loss)
        #     plt.show()
