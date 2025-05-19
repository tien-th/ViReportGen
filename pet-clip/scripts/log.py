# Create a logger class similar to the second code's nnUNetLogger
import wandb 

class Logger:
# Recording training losses ('train_losses')
# Tracking the start and end times of epochs ('epoch_start_timestamps' and 'epoch_end_timestamps')
# Calculating epoch duration for performance monitoring
    def __init__(self, project, experiment_name=None):
        wandb.login(key = '9ab49432fdba1dc80b8e9b71d7faca7e8b324e3e')
        wandb.init(project=project, name=experiment_name)
        self.experiment_name = experiment_name
        self.logs_dict = {}

    def log(self, key, value):
            # wandb.log({key: value})
        self.logs_dict[key] = value
    
    def log_img(self, key, img_tensor, section='train'):
        # img_tensor: shape [1, h, w] 
        # log the img to wandb 
        wandb.log({f'{section}/{key}': wandb.Image(img_tensor)})

    def log_validation_loss(self, loss, section='valid'):
        wandb.log({f'{section}/loss': loss})

    def log_to_wandb(self, section='train'):
        wandb.log({f'{section}/{key}': value for key, value in self.logs_dict.items()})
        self.logs_dict = {}