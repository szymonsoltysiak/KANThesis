import torch
import torch.nn as nn
import numpy as np
from KANLayer import KANLayer
import os 
import random
from tqdm import tqdm

class KAN(nn.Module):
    def __init__(self, width=None, grid=3, k=3, 
                 noise_scale=0.1, noise_scale_base=0.1, 
                 base_fun=torch.nn.SiLU(), bias_trainable=True, 
                 grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, 
                 sb_trainable=True, device='cpu', seed=0):
        super(KAN, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.biases = []
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width

        for i in range(self.depth):
            scale_base = 1 / np.sqrt(width[i]) + (torch.randn(width[i] * width[i + 1], ) * 2 - 1) * noise_scale_base
            sp_batch = KANLayer(in_dim=width[i], out_dim=width[i + 1], num=grid, k=k, noise_scale=noise_scale, 
                                scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, 
                                grid_range=grid_range, sp_trainable=sp_trainable,
                                sb_trainable=sb_trainable, device=device)
            self.act_fun.append(sp_batch)

            bias = nn.Linear(width[i + 1], 1, bias=False, device=device).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        self.device = device

    def update_grid_from_samples(self, x):
        for l in range(self.depth):
            self.forward(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])   

    def forward(self, x):
        self.acts = [] 
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []

        self.acts.append(x) 

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            x = x_numerical 
            postacts = postacts_numerical

            grid_reshape = self.act_fun[l].grid.reshape(self.width[l + 1], self.width[l], -1)
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)

        return x
    
    def train(self, dataset, steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., 
              lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., 
              stop_grid_update_step=50, batch=-1, small_mag_threshold=1e-16, small_reg_factor=1., 
              metrics=None, sglr_avoid=False, device='cpu'):      
        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy

            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg(self.acts_scale)
            loss = train_loss + lamb * reg_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            if _ % log == 0:
                pbar.set_description("train loss: %.2e | test loss: %.2e | reg: %.2e " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

        return results

