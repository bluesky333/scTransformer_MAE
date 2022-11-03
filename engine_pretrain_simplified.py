import torch



def train_one_epoch(model: torch.nn.Module,
                    data_loader: torch.utils.data.Dataloader, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int,
                    log_writer=None,
                    args=None):
    model.train()

    for idx, batch in enumerate(data_loader):
        pass
