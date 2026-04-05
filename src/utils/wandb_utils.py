import os


class WandbConfig:
    def __init__(self):
        self.enabled = True
        self.project = 'face-spoofing-detection'
        self.entity = None
        
        self.log_model = True
        self.log_freq = 10
        
        self.tags = ['ensemble', 'face-spoofing', 'open-set']
        
        self.watch_model = True
        self.watch_log = 'all'
        self.watch_log_freq = 100


def get_wandb_config():
    return WandbConfig()


import os


def init_wandb(config, run_name, model_type, notes=None):
    if not config.enabled:
        return None
    
    import wandb
    
    api_key = os.getenv('WANDB_API_KEY')
    if api_key:
        wandb.login(key=api_key)
    
    run = wandb.init(
        project=config.project,
        entity=config.entity,
        name=run_name,
        tags=config.tags + [model_type],
        notes=notes,
        config={
            'model_type': model_type,
        }
    )
    
    return run


def log_metrics(metrics_dict, step=None):
    import wandb
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)


def finish_wandb():
    import wandb
    if wandb.run is not None:
        wandb.finish()
