import torch
import torchvision
from torchvision.transforms import transforms
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchmetrics

pl.seed_everything(0)

class StanfordCarsNet(pl.LightningModule):

    def __init__(self, lr=1e-2, weight_decay=1e-4, is_finetuned = False):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.is_finetuned = is_finetuned

        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_features = backbone.fc.in_features
        num_target_classes = 196
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_target_classes)
        )


        # Define metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass",
                                               num_classes=num_target_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=num_target_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass",
                                             num_classes=num_target_classes)

    def forward(self, x):
        if self.is_finetuned:
            representations = self.feature_extractor(x).flatten(1)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)

        logits = self.classifier(representations)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_acc(logits, y)

        tensorboard_logs = {"train_loss":loss, "train_acc": self.train_acc}
        self.log_dict(tensorboard_logs, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(logits, y)

        tensorboard_logs = {"val_loss":loss, "val_acc":self.val_acc}
        self.log_dict(tensorboard_logs, prog_bar=True, on_step=False, on_epoch=True)


    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return preds


    def configure_optimizers(self):
        # TODO: add learning rate scheduler
        optimizer = optim.Adam(self.parameters(),
                               lr=self.learning_rate,
                               weight_decay = self.weight_decay)


        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                            factor=0.5,
                                                            verbose=True,
                                                            cooldown=0)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }

        return [optimizer], [lr_scheduler_config]

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc(logits, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        
