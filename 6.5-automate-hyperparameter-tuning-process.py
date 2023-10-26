from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import ModelCheckpoint
from common_def import LightningModel, PyTorchMLP, CustomDataModule

if __name__ == '__main__':

    cli = LightningCLI(
        model_class=LightningModel,
        datamodule_class=CustomDataModule,
        run=False,
        seed_everything_default=123,
        trainer_defaults={
            'max_epochs': 10,
            'accelerator': 'cpu',
            'callbacks': [ModelCheckpoint(monitor='val_acc', mode='max')]
        }
    )

    model = LightningModel(
        num_features=100, 
        num_classes=2, 
        hidden_units=cli.model.hidden_units,
        learning_rate=cli.model.learning_rate)
    
    cli.trainer.fit(model=model, datamodule=cli.datamodule)