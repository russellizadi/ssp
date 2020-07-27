import torch
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer


class KFAC(Optimizer):

    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.alpha = .1
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        #self.lam = lam

        for mod in net.modules():
            mod_name = mod.__class__.__name__
            if mod_name in ['CRD', 'CLS']:
                #print(mod)
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                handle = mod.register_backward_hook(self._save_grad_output)
                self._bwd_handles.append(handle)
                for sub_mod in mod.modules():
                    i_sub_mod = 0
                    if hasattr(sub_mod, 'weight'):
                        assert i_sub_mod == 0
                        params = [sub_mod.weight]
                        if sub_mod.bias is not None:
                            params.append(sub_mod.bias)

                        d = {'params': params, 'mod': mod, 'sub_mod': sub_mod}
                        self.params.append(d)
                        i_sub_mod += 1

        super(KFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True, lam=0.):
        """Performs one step of preconditioning."""
        self.lam = lam
        fisher_norm = 0.
        for group in self.param_groups:
            
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)

            if update_params:
                # Preconditionning
                #print(weight.grad.data.shape, weight.grad.data)
                #print(bias.grad.data.shape, bias.grad.data)
                gw, gb = self._precond(weight, bias, group, state)
                #print(gw.shape, gw)
                #print(gb.shape, gb)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                    #print(fisher_norm)
                    #print(adf)
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
                    
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            #print(scale)
            for group in self.param_groups:
                for param in group['params']:
                    print(param.shape, param)
                    param.grad.data *= scale
            #print(adfa)
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        # i: (x, edge_index)
        if mod.training:
            self.state[mod]['x'] = i[0]
            self.mask = i[-1]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(1)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        ixxt = state['ixxt'] # [d_in x d_in]
        iggt = state['iggt'] # [d_out x d_out]
        g = weight.grad.data # [d_in x d_out]
        s = g.shape
        
        # new
        g = g.contiguous().view(-1, g.shape[-1])
            
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(1, gb.shape[0])], dim=0)

        
        #print(ixxt.shape, g.shape, iggt.shape)
        g = torch.mm(ixxt, torch.mm(g, iggt))
        if bias is not None:
            gb = g[-1].contiguous().view(*bias.shape)
            g = g[:-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        #mod = group['mod']
        sub_mod = group['sub_mod']
        x = self.state[group['mod']]['x'] # [d_in x n]
        gy = self.state[group['mod']]['gy'] # [n x d_out]
        
        n = float(self.mask.sum() + self.lam*((~self.mask).sum()))
        #print(n)
        # Computation of xxt
        #x = x.data[self.mask].t() # [d_in x n]
        x = x.data.t()
        
        # new
        if sub_mod.weight.ndim == 3:
            x = x.repeat(sub_mod.weight.shape[0], 1)
        #print(x.shape)
        
        if sub_mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        #x -= x.mean(1, keepdim=True)
        #x[:, self.mask] *= self.lam
        #x[:, ~self.mask] *= 1. - self.lam


        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / n #float(x.shape[1]) #float(self.mask.sum()) #
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)

        
        #print(state['xxt'].shape, state['xxt'])
        
        # Computation of ggt
        #gy = gy.data[self.mask].t() # [d_out x n]
        gy = gy.data.t() # [d_out x n]
        #gy -= gy.mean(1, keepdim=True)
        #gy[:, self.mask] *= self.lam**.5
        #gy[:, ~self.mask] *= 1. - self.lam**.5


        state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / n #float(gy.shape[1])#float(self.mask.sum()) #
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)
        #print(state['ggt'].shape, state['ggt'])
        
        #print(adsf)

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()

        return ixxt, iggt

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()