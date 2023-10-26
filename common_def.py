from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
import lightning
import torchmetrics
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, dataset
import torch.nn.functional as F
from sklearn import datasets, model_selection

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes=2, hidden_units=[50, 25]):
        super().__init__()

        hidden_layers = []
        for hidden_unit in hidden_units:
            hidden_layers.append(torch.nn.Linear(num_features, hidden_unit))
            # batch normalization: why it works?
            hidden_layers.append(torch.nn.BatchNorm1d(hidden_unit))
            # activation function
            hidden_layers.append(torch.nn.ReLU())
            # DROPOUT: why it works?
            # hidden_layers.append(torch.nn.Dropout(0.2))
            
            num_features = hidden_unit

        output_layer = torch.nn.Linear(hidden_units[-1], num_classes)
        hidden_layers.append(output_layer)

        self.layers = torch.nn.Sequential(*hidden_layers)

        # layer1_outputs = 50
        # layer2_outputs = 25
        
        # self.layers = torch.nn.Sequential(
        #     torch.nn.Linear(num_features, layer1_outputs),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(layer1_outputs, layer2_outputs),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(layer2_outputs, num_classes)
        # )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.layers(x)
        return logits


class LightningModel(lightning.LightningModule):
    def __init__(self, torch_model=None, num_classes=None, num_features=None, hidden_units=None, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate

        if torch_model is not None:
            self.torch_model = torch_model
        else:
            self.torch_model = PyTorchMLP(num_features=num_features, hidden_units=hidden_units, num_classes=num_classes)

        # save hyperparameters (but skip the model parameters)
        self.save_hyperparameters(ignore=['model'])

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.torch_model(x)
    
    def _process_step(self, batch, batch_idx):
        batch_features, batch_labels = batch
        logits = self.forward(batch_features)
        loss = F.cross_entropy(logits, batch_labels)
        predictions = torch.argmax(logits, dim=1)
        return loss, predictions, batch_labels
    
    def training_step(self, batch, batch_idx):
        loss, predictions, labels = self._process_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.train_acc(predictions, labels)
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._process_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.val_acc(predictions, labels)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, predictions, labels = self._process_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.val_acc(predictions, labels)
        self.log('test_acc', self.val_acc, prog_bar=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        ## without scheduler
        return optimizer

        ## use Learning Rate scheduler: for every 10 epochs, reduce the learning rate by a factor of 0.5
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        # return [optimizer], [scheduler]
    

class CustomDataset(Dataset):
    def __init__(self, X_features, y_labels, transform=None):
        self.features = X_features
        self.labels = y_labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    

class CustomDataModule(lightning.LightningDataModule):
    def __init__(self, data_dir='./dataset/', batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # generate examples
        X, y = datasets.make_classification(
            n_samples=20000,
            n_features=100,
            n_classes=2,
            n_informative=10,
            n_redundant=40,
            n_repeated=25,
            n_clusters_per_class=5,
            flip_y=0.05,
            class_sep=0.5,
            random_state=123,
        )
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=123)
        X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=123)
        
        self.train_dataset = CustomDataset(X_train.astype(np.float32), y_train.astype(np.int64))
        self.val_dataset = CustomDataset(X_val.astype(np.float32), y_val.astype(np.int64))
        self.test_dataset = CustomDataset(X_test.astype(np.float32), y_test.astype(np.int64))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def plot_metrics_from_csv_log(path):
    df_metrics = pd.read_csv(path)
    df_metrics.head()
    df_metrics_aggr = df_metrics.groupby(['epoch']).mean()

    df_metrics_aggr[['train_loss', 'val_loss']].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Loss')
    df_metrics_aggr[['train_acc', 'val_acc']].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Accuracy')
    plt.show()