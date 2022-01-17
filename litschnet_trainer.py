from torch.data.utils import DataLoader

class LitSchNetTrainer() :
    
    def __init__(litschnet, 
                 optimizer, 
                 scheduler, 
                 early_stopping:bool=False) :
        self.litschnet = litschnet
        self.litschnet.to('cuda')
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.writer = 
        
    def dataset_iteration(datasets, train=False) :
        for dataset in datasets :
            loader = DataLoader(dataset, 
                                batch_size=64, 
                                shuffle=True, 
                                num_workers=4)
            for batch_idx, batch in (enumerate(loader)) :
                batch.to('cuda')
                if train : 
                    optimizer.zero_grad()
                    
                loss = litschnet.training_step(batch, batch_idx)
                
                if train :
                    loss.backward()
                    losses.append(loss.detach().cpu().numpy().item())
                    optimizer.step()
                else :
                    losses.append(loss.item())
                    
    def train(model, train_datasets, val_datasets, max_epoch:int=50) :
        for epoch in range(max_epoch) :
            