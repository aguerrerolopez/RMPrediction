#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:26:30 2020

@author: root
"""

# -- coding: utf-8 --
# -- coding: utf-8 --
import torch
from torch import nn
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import torch.distributions as td
import numpy as np


class NeuralNetworks(nn.Module):
    def __init__(self, dim, in_dim, out_dim, mean=False, cov=False,
                 tgx=False) -> object:
        """
        Class that creates the three neural networks needed to do the inference. The booleans chooses
        which one of the three NNs (mu, sigma or t) are you going to create
        :param dim: dict. First entry is a list of len equal to number of layers of mu and sigma NNs with the number of
        neurons per layer. Second entry is a list of len equal to number of layers of t NN with the the number of
        neurons per layer. Third entry is a list of three floats with the learning rate of each NN.
        :param in_dim: int. Input's dimension of the NN.
        :param out_dim: int. Output's dimension of the NN.
        :param mean: bool. If true, the mu NNs is going to be created.
        :param cov: bool. If true, the sigma NNs is going to be created.
        :param tgx: bool. If true, the t NNs is going to be created.
        """
        super().__init__()

        self.train_mean = mean
        self.train_cov = cov
        self.train_tgx = tgx
        # Common layers for both mu and sigma.
        if self.train_mean or self.train_cov:
            # self.hidden_1 = nn.Linear(in_dim, dim['xga'][0])
            # self.hidden_2 = nn.Linear(dim['xga'][0], dim['xga'][1])
            # self.hidden_3 = nn.Linear(dim['xga'][1], dim['xga'][2])
            # self.output = nn.Linear(dim['xga'][-1], out_dim)
            self.output = nn.Linear(in_dim, out_dim)
        # Output activation for sigma.
        if self.train_cov:
            self.out_act = nn.Softplus()
        # Layers for t.
        if self.train_tgx:
            self.hidden_1 = nn.Linear(in_dim, dim['tgx'][0])
            # self.hidden_2 = nn.Linear(dim['tgx'][0], dim['tgx'][1])
            # self.hidden_3 = nn.Linear(dim['tgx'][1], dim['tgx'][2])
            # self.output = nn.Linear(dim['tgx'][-1], out_dim)
            self.output = nn.Linear(in_dim, out_dim)
            self.out_act = nn.LogSoftmax(dim=1)

        # Common activation after each hidden layer
        self.activation = nn.ReLU()
        self.activation2 = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, input_vector):

        if self.train_mean:
            # Layer 1
            # self.mu = self.activation(self.hidden_1(input_vector))
            # Layer 2
            # self.mu = self.activation(self.hidden_2(self.mu))
            # # Layer 3
            # self.mu = self.activation(self.hidden_3(self.mu))
            # Output layer
            # self.mu = self.dropout(self.mu)
            # self.mu = self.output(self.mu)
            self.mu = self.output(input_vector)
        if self.train_cov:
            # Layer 1
            # self.sigma = self.activation(self.hidden_1(input_vector))
            # # Layer 2
            # self.sigma = self.activation(self.hidden_2(self.sigma))
            # # Layer 3
            # self.sigma = self.activation2(self.hidden_3(self.sigma))
            # Output layer
            # self.sigma = self.dropout(self.sigma)
            # self.sigma = self.out_act(self.output(self.sigma))
            self.sigma = self.out_act(self.output(input_vector))
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
    def __init__(self, d_x, dim, t_raw=None, t_ohe=None, store=True):
        """
        Class that implements the module CategoricalNN. Given a Z, W and t it returns the approximation to X that
        explains the t categories.
        :param d_x: int. Output's dimension of the approximation of X.
        :param dim: dict. First entry is a list of len equal to number of layers of mu and sigma NNs with the number of
        neurons per layer. Second entry is a list of len equal to number of layers of t NN with the the number of
        neurons per layer. Third entry is a list of three floats with the learning rate of each NN.
        :param t_raw: numpy array. Array of shape 1xN with the category that it belongs each n-point.
        :param t_ohe: numpy array. Array of shape NxC with the one-hot encoder version of t_raw. If None it will be computed.
        """
        # Looks for GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training categorical using: ", self.device)

        if t_ohe is None:
            self.t_ohe = torch.from_numpy(LabelBinarizer().fit_transform(t_raw)).double()
        if t_raw is None:
            self.t_ohe = torch.from_numpy(t_ohe)

        self.t_ohe.to(self.device)

        # Define in and out dimensions of each NN
        self.dim_x = [d_x + self.t_ohe.shape[1], d_x]
        self.dim_t = [d_x, self.t_ohe.shape[1]]
        # Instance the three NNs
        self.mu_nn = NeuralNetworks(dim=dim, in_dim=self.dim_x[0], out_dim=self.dim_x[1], mean=True)
        self.sigma_nn = NeuralNetworks(dim=dim, in_dim=self.dim_x[0], out_dim=self.dim_x[1], cov=True)
        self.tgx_nn = NeuralNetworks(dim=dim, in_dim=self.dim_t[0], out_dim=self.dim_t[1], tgx=True)
        # Instance the optimizers of the NNs
        self.opt_mean = torch.optim.Adam(self.mu_nn.parameters(), dim['lr'][0])
        self.opt_cov = torch.optim.Adam(self.sigma_nn.parameters(), dim['lr'][1])
        self.opt_tgx = torch.optim.Adam(self.tgx_nn.parameters(), dim['lr'][2])
        # Move the NNs to GPU if exists
        self.tgx_nn.to(self.device)
        self.mu_nn.to(self.device)
        self.sigma_nn.to(self.device)
        self.nlloss = nn.NLLLoss(reduction="sum")

        self.store = store


    def old_montecarlo(self, iterations=10):
        accumulative_loss = 0
        for i in range(iterations):
            # Get a sample of X
            x_aprox = self.mu_nn.mu + torch.empty(1).normal_().to(self.device) * torch.sqrt(self.sigma_nn.sigma)
            # Forward
            self.tgx_nn.forward(x_aprox)
            _, targets = self.t_ohe.max(dim=1)
            accumulative_loss += -self.nlloss(self.tgx_nn.tgx, targets.to(self.device))
        return accumulative_loss / iterations

    def elbo(self, z, w, tau, y, nn_in, lambda_x=1, it=1):

        # Set the zerograd
        self.opt_mean.zero_grad()
        self.opt_cov.zero_grad()
        self.opt_tgx.zero_grad()

        # Forward to mu and sigma
        self.mu_nn.forward(nn_in)
        self.sigma_nn.forward(nn_in)

        # E[T | X]
        exp_tgx = self.old_montecarlo(iterations=it)
        # E[X | W, Z, tau]
        exp_xga = -tau * 0.5 * (torch.sum(self.sigma_nn.sigma) +
                                torch.sum(torch.pow(self.mu_nn.mu, 2))) + tau * torch.sum(
            torch.diag(self.mu_nn.mu @ w.float() @ z.float().T))

        # H(q(X))
        entropy_xga = 0.5 * torch.sum(torch.log(self.sigma_nn.sigma))

        # Construct the loss
        loss = -1 * (exp_tgx + exp_xga + entropy_xga)

        # Calculate gradients
        loss.backward(retain_graph=True)
        # Optimize it
        self.opt_tgx.step()
        self.opt_mean.step()
        self.opt_cov.step()

        return -exp_tgx, -exp_xga, -entropy_xga, loss

    def aprox(self, numpy=True):
        if numpy:
            return self.mu_nn.mu.data.cpu().numpy(), self.sigma_nn.sigma.data.cpu().numpy()
        else:
            return self.mu_nn.mu.data.cpu(), self.sigma_nn.sigma.data.cpu()

    def predict_proba(self, Z, W, b=None, numpy=False, samples=5):
        with torch.no_grad():
            self.tgx_nn.eval()
            predict = torch.zeros(Z.shape[0], self.dim_t[1], samples)
            for i in range(samples):
                if b is not None:
                    x_aprox = Z.float()@W.float().T + b.float()
                else:
                    x_aprox = Z.float()@W.float().T
                self.tgx_nn.forward(x_aprox.to(self.device))
                predict[:, :, i] = torch.exp(self.tgx_nn.tgx.data.cpu())
            return torch.mean(predict, dim=2).numpy()

    def predict(self, X_in, numpy=False):
        with torch.no_grad():
            self.tgx_nn.eval()
            self.tgx_nn.forward(X_in.to(self.device))
            output = np.exp(self.tgx_nn.tgx.data.cpu().numpy())
            prediction = np.argmax(output, axis=1)
            # prediction = torch.argmax(torch.exp(self.tgx_nn.tgx.data.cpu()), 1).cpu().numpy()
            return prediction

    def SGD_step(self, z_mean, w_mean, tau, steps=500, batch_size=128, mc_it=1,
                lambda_x=1, print_every=20):

        # Random variables initialization
        tau = tau
        z = torch.from_numpy(z_mean).to(self.device)
        w = torch.from_numpy(w_mean).to(self.device)
        y = (z @ w.T).to(self.device)
        # Input of mean and cov NNs
        nn_in = torch.cat((y.double().to(self.device), self.t_ohe.double().to(self.device)), 1).float()

        # Train mode
        self.mu_nn.train()
        self.sigma_nn.train()
        self.tgx_nn.train()
        print("Train mode")

        if self.store:
            full_loss = []
            tgx_loss = []
            xga_loss = []
            entropy_loss = []

        for i in range(steps):
            if i % print_every == 0:
                print('\rSteps performed: %d/%d' % (i, steps), end='\r',
                             flush=True)

            tgx_i, xga_i, entropy_i, loss = self.elbo(z=z, w=w, tau=tau, nn_in=nn_in, y=y, lambda_x=lambda_x, it=mc_it)

            if self.store:
                tgx_loss.append(tgx_i.data.cpu().numpy())
                xga_loss.append(xga_i.data.cpu().numpy())
                entropy_loss.append(entropy_i.data.cpu().numpy())
                full_loss.append(-loss.data.cpu().numpy())

        if self.store:
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.title("E[t|X]")
            plt.plot(tgx_loss)

            plt.subplot(2, 2, 2)
            plt.title("E[X|W, Z]")
            plt.plot(xga_loss)

            plt.subplot(2, 2, 3)
            plt.title("H(q(x))")
            plt.plot(entropy_loss)

            plt.subplot(2, 2, 4)
            plt.title("Categorical ELBO")
            plt.xlabel("Iterations")
            plt.ylabel("LB")
            plt.plot(full_loss)
            plt.show()



