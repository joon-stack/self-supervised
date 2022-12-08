import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
import torch.backends.cudnn as cudnn
import wandb
from tqdm import trange, tqdm
from datasets import get_ds
from cfg import get_cfg
from methods import get_method
from eval.get_data import get_data


def get_scheduler(optimizer, cfg):
    if cfg.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.epoch if cfg.T0 is None else cfg.T0,
            T_mult=cfg.Tmult,
            eta_min=cfg.eta_min,
        )
    elif cfg.lr_step == "step":
        m = [cfg.epoch - a for a in cfg.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=cfg.drop_gamma)
    else:
        return None


class Model(nn.Module):
    def __init__(self, cfg, num_classes=100):
        super().__init__()
        load_model = get_method(cfg.method)(cfg)
        if cfg.fname is None:
            print("evaluating random model")
        else:
            load_model.load_state_dict(torch.load(cfg.fname))

        # get only main model (not header)
        self.feat_model = load_model.model
        self.feat_size = load_model.out_size
        self.num_classes = num_classes

        ###################
        # init new header #
        ###################
        #
        #
        #
        #
        #


    def forward(self, x):
        #######################################################
        # extract feature and obtain logit through new header #
        #######################################################
        #
        #
        #
        #
        #

    def get_acc(self, ds_test, topk=[1, 5]):
        self.eval()
        predict_test, target_test = get_data(self, ds_test, self.num_classes, "cuda")
        predict_test = predict_test.topk(max(topk), 1, largest=True, sorted=True).indices
        acc = {t: (predict_test[:, :t] == target_test[..., None]).float().sum(1).mean().cpu().item()
               for t in topk}
        return acc


if __name__ == "__main__":
    cfg = get_cfg()
    wandb.init(project=cfg.wandb, config=cfg)

    # get model
    cudnn.benchmark = True
    model = Model(cfg=cfg).to('cuda')

    # get dataset
    ds = get_ds(cfg.dataset)(cfg.bs, cfg, cfg.num_workers)

    # get loss fn
    loss_fn = nn.CrossEntropyLoss()

    # get optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_l2)

    # get lr scheduler
    scheduler = get_scheduler(optimizer=optimizer, cfg=cfg)

    # eval interval
    eval_every = cfg.eval_every

    # for each epoch
    for ep in trange(cfg.epoch, position=0):
        loss_ep = []
        iters = len(ds.train)
        for n_iter, (x_data, y_data) in enumerate(tqdm(ds.train, position=1)):
            optimizer.zero_grad()
            loss = loss_fn(model(x_data[0].to('cuda')), y_data.to('cuda'))
            loss.backward()
            optimizer.step()

            loss_ep.append(loss.item())
            if cfg.lr_step == "cos":
                scheduler.step(ep + n_iter / iters)

        if cfg.lr_step == "step":
            scheduler.step()

        if len(cfg.drop) and ep == (cfg.epoch - cfg.drop[0]):
            eval_every = cfg.eval_every_drop

        if (ep + 1) % eval_every == 0:
            acc = model.get_acc(ds.test)
            wandb.log({"acc": acc[1], "acc_5": acc[5]}, commit=False)

        if (ep + 1) % 100 == 0:
            fname = f"data/transfer_learning_{cfg.method}_{cfg.dataset}_{ep}.pt"
            torch.save(model.state_dict(), fname)

        wandb.log({"loss": np.mean(loss_ep), "ep": ep})
