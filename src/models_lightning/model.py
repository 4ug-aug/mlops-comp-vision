import timm
import torch
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy


class MyAwesomeModel(LightningModule):
    def __init__(
        self,
        model_name="resnet18",
        classes=10,
        lr=1e-2,
        weight_decay=0,
        batch_size=64,
        optimizer="adam",
        dataset_path="data/processed/",
        num_workers=1,
        pretrained=False,
    ):

        super().__init__()

        self.model_name = model_name
        self.classes = classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.dataset_path = dataset_path
        self.num_workers = num_workers
        self.pretrained = pretrained

        self.cnn = timm.create_model(
            self.model_name, pretrained=self.pretrained, num_classes=self.classes
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.classes)
        return loss, acc

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            opt = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return opt

    def train_dataloader(self):
        trainset = torch.load(f"{self.dataset_path}train.pt")
        return DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        testset = torch.load(f"{self.dataset_path}test.pt")
        return DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        valset = torch.load(f"{self.dataset_path}val.pt")
        return DataLoader(
            valset, batch_size=self.batch_size, num_workers=self.num_workers
        )
