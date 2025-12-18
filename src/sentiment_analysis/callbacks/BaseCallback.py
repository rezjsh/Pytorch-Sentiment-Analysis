
class BaseCallback:
    """Base class for defining the callback interface."""
    def on_train_start(self, trainer): pass
    def on_epoch_start(self, trainer, epoch): pass
    def on_batch_end(self, trainer, batch_index, loss): pass
    def on_epoch_end(self, trainer, epoch, val_loss): pass
    def on_train_end(self, trainer): pass