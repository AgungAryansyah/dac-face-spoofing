class EnsembleConfig:
    def __init__(self):
        self.num_classes = 5
        self.image_size = 224
        
        self.dino_checkpoint = 'outputs/checkpoints/dino/best_model.pth'
        self.vit_checkpoint = 'outputs/checkpoints/vit/best_model.pth'
        self.resnet_checkpoint = 'outputs/checkpoints/resnet/best_model.pth'
        self.wresnet_checkpoint = 'outputs/checkpoints/wide_resnet/best_model.pth'
        
        self.feature_dim = 768 + 768 + 2048 + 2048
        
        self.xgboost_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42
        }
        
        self.msp_threshold = 0.5
        self.unknown_class_label = 'fake_unknown'
        
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        
        self.features_cache_dir = 'outputs/features'
        self.model_save_path = 'model/xgboost_ensemble.pkl'
        
        self.random_seed = 42


def get_config():
    return EnsembleConfig()
