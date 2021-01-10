import wandb
import models
import models

def initialize_wandb():
    wandb.init(project="fed-learning")
    # wandb.watch(model)
    return wandb.log

# w = initialize_wandb(models.TwoLayerNet)