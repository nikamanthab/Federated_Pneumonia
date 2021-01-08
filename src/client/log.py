import wandb
import models
import models

def initialize_wandb(model, node_name):
    wandb.init(project=node_name)
    wandb.watch(model)
    return wandb.log

# w = initialize_wandb(models.TwoLayerNet)