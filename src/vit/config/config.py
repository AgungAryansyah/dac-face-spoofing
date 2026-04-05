import torch


class VitConfig:
    def __init__(self):
        self.model_name = 'vit_base_patch16_224'
        self.num_classes = 5
        self.image_size = 224
        self.feature_dim = 768
        
        self.batch_size = 32
        self.num_epochs = 30
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.lr_scheduler = 'cosine'
        self.warmup_epochs = 3
        
        self.freeze_ratio = 0.75
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 4
        self.pin_memory = True
        
        self.checkpoint_dir = 'outputs/checkpoints/vit'
        self.log_dir = 'outputs/logs/vit'
        
        self.save_best_only = True
        self.early_stopping_patience = 5
        
        self.random_seed = 42


def get_config():
    return VitConfig()
