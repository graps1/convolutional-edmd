import torch
import tqdm

class Observer(torch.nn.Module):

    def __init__(self, n_obs=10):
        super().__init__()
        self.n_obs = n_obs
        self.conv = torch.nn.LazyConv2d(n_obs, 1)
        self.nl = torch.nn.Softplus()

    def forward(self, x):
        x = self.conv(x)
        x = self.nl(x)
        return x


class Reconstructor(torch.nn.Module):

    def __init__(self, obsC, C):
        super().__init__()
        self.n_obs = torch.tensor(obsC.shape[1])
        self.conv = torch.nn.Conv2d(self.n_obs, C.shape[1], 1)
        self.train(obsC, C)

    def train(self, obsC, C):
        torch.set_grad_enabled(True)
        opt = torch.optim.Adam(self.parameters(), lr=1e-1)
        for _ in ( pbar := tqdm.tqdm(range(100))):
            opt.zero_grad()
            loss = (self(obsC) - C).pow(2).mean()
            loss.backward()
            opt.step()
            pbar.set_description(f"loss: {loss.item():.3e}")
        torch.set_grad_enabled(False)

    def forward(self, x):
        x = self.conv(x)
        return x