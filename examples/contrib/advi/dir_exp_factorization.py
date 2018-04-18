import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.contrib.autoguide import (AutoDiagonalNormal, AutoMultivariateNormal,
                                    AutoGuideList)
import pyro.optim as optim
from utils import get_data


def model(K, U, I, c, y):
    assert y.shape == (U, I)
    theta = pyro.sample("theta", dist.Dirichlet(1000. * torch.ones(K, U)).independent(1))
    beta = pyro.sample("beta", dist.Exponential(c * torch.ones(K, I)).independent(1))
    with pyro.iarange('data', len(y)):
        pyro.sample('obs', dist.Poisson(torch.exp(torch.mm(theta.t(), beta)))
                    .independent(1), obs=y.float())


def main(args):
    guide = AutoGuideList(model)
    adam = optim.Adam({'lr': 1e-3})
    svi = SVI(model, guide, adam, loss="ELBO", enum_discrete=True, max_iarange_nesting=1)
    for i in range(100):
        loss = svi.step(*args)
        print('loss=', loss)
        if i % 10 == 9:
            d = guide.parts[0].median()
            print({k: d[k].sum() / 200 for k in ["theta", "beta"]})


if __name__ == "__main__":
    varnames = ['K', 'U', 'I', 'c', 'y']
    args = get_data("../../data/dir_exp_training_data.json", varnames)
    main(args)
