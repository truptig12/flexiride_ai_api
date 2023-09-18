
# steps to learn - how to add learning rate finder and then how to load saved weights from files on API side need to create classes for all thee models, how to use that method, how to send response in JSON, change image into respective formate for ai models, run predications,
log_dir = "/content/"


# !pip install pytorch-lightning
# !pip install torchmetrics

import torch
import torchvision
from torchvision.transforms import transforms
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
import pytorch_lightning as pl
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import torchmetrics
import sklearn
import os
from torchvision.transforms.functional import rotate
from torchvision.datasets import StanfordCars
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

pl.seed_everything(0)


train_data = StanfordCars(root="/content/",split="train",download=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Resize((224,224))
                                               ]))
loader = DataLoader(train_data, batch_size= 64, shuffle=False, num_workers=1)

mean = 0.
std = 0.
nb_samples = 0.
for data, _ in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples



# For stanford cars dataset
mean = torch.tensor(mean)
std = torch.tensor(std)

train_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.Normalize(mean, std),
                                      transforms.RandomAdjustSharpness(sharpness_factor=2),
                                      transforms.RandomAutocontrast(),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomAffine(degrees=35, translate=(0.3, 0.3)),
                                      ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224)),
                                     transforms.Normalize(mean, std)])

"""## Create Train, Val and Test Dataloaders
Splitting the data in the training folder into training_data and val_data. Keeping the data in the test folder entirely for test_data. For the split I am using a 80:20 split because the number for images for training is less than 10,000.
"""

train_data = StanfordCars(root="/content/",
                                           split="train",
                                           download=True,
                                           transform=train_transform)

val_data = StanfordCars(root="/content/",
                                         split="train",
                                         download=True,
                                         transform=test_transform)

val_count = round(len(train_data) * 0.2)
train_count = len(train_data) - val_count

train_data, _ = torch.utils.data.random_split(train_data, [train_count, val_count])
_, val_data = torch.utils.data.random_split(val_data, [train_count, val_count])


test_data = StanfordCars(root="/content/",
                                          split="test",
                                          download=True,
                                          transform = test_transform)


classes = train_data.dataset.classes

batch_size = 128  # TODO: add automatic batch size finder for PL

train_dataloader = DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True)


val_dataloader = DataLoader(val_data,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 2)


test_dataloader = DataLoader(test_data,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             num_workers = 2)


""" Define LightningModule"""

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

model = StanfordCarsNet(is_finetuned = False)


"""## Setup Callbacks"""

## Metric tracker

class MetricTracker(Callback):

    def __init__(self):
        self.val_loss = []
        self.val_acc = []
        self.train_loss = []
        self.train_acc = []
        self.lr = []

    def on_validation_epoch_end(self, trainer, module):
        val_loss = trainer.logged_metrics['val_loss'].item()
        val_acc = trainer.logged_metrics['val_acc'].item()
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)

    def on_train_epoch_end(self, trainer, module):
        train_loss = trainer.logged_metrics['train_loss'].item()
        train_acc = trainer.logged_metrics['train_acc'].item()
        lr = module.optimizers().param_groups[0]['lr']
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.lr.append(lr)

metric_tracker = MetricTracker()

"""This is used to save model weights based on the val_loss metric."""


checkpoint_callback = ModelCheckpoint(monitor='val_loss')

"""This is used for logging the learning rate so that we can later review it."""

lr_monitor = LearningRateMonitor(logging_interval='epoch')

"""## Train"""

MAX_EPOCHS = 75
lr = 0.003311311214825908
weight_decay=1e-4


model = StanfordCarsNet(lr=lr, weight_decay=weight_decay, is_finetuned = False)

trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                     default_root_dir = os.path.join(log_dir, "phase1"),
                     callbacks=[metric_tracker, checkpoint_callback, lr_monitor])

trainer.fit(model=model,
            train_dataloaders = train_dataloader,
            val_dataloaders = val_dataloader)

"""## Training Visualization"""

sns.lineplot(x=range(len(metric_tracker.train_loss)), y=metric_tracker.train_loss, label="train_loss")
sns.lineplot(x=range(len(metric_tracker.val_loss)), y=metric_tracker.val_loss, label="val_loss")
plt.legend()
plt.show()

sns.lineplot(x=range(len(metric_tracker.train_acc)), y=metric_tracker.train_acc, label="train_acc")
sns.lineplot(x=range(len(metric_tracker.val_acc)), y=metric_tracker.val_acc, label="val_acc")
plt.legend()
plt.show()

"""Around the 45th epoch, learning rate was halved. This allowed the training and validation loss to continue decreasing again."""

sns.lineplot(x=range(len(metric_tracker.lr)), y=metric_tracker.lr, label="lr")
plt.legend()
plt.show()

"""## Test
The best model stored by the model checkpoint callback is automatically loaded during running the test step of lightning.
"""

print("Best model path: " + checkpoint_callback.best_model_path)

trainer.test(model, test_dataloader)

"""# Fine Tune
In this section, the backbone feature_extractor is fine tuned for the current dataset.

## Load Model Weights
The model weights corresponding to the lowest validation loss from the previous section is loaded into the model. This means the backbone is loaded with pre-trained ResNet50 weights and the classification layer is loaded with the best weight obtained during the last phase of training.
"""

checkpoint_callback.best_model_path

model = StanfordCarsNet.load_from_checkpoint(checkpoint_callback.best_model_path, is_finetuned = True)

dev_trainer = pl.Trainer(fast_dev_run=True, accelerator="gpu")
dev_trainer.fit(model, train_dataloader, val_dataloader)


"""## Train
We keep the weight decay same as before but we change the learning rate and number of epochs.
"""

MAX_EPOCHS = 75
lr = 0.0002089296130854041
# weight_decay=1e-4


model = StanfordCarsNet.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    lr=lr,
    weight_decay=weight_decay,
    is_finetuned = True)

trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                     accelerator='gpu',
                     default_root_dir = os.path.join(log_dir, "phase2"),
                     callbacks=[metric_tracker, checkpoint_callback, lr_monitor])

trainer.fit(model=model,
            train_dataloaders = train_dataloader,
            val_dataloaders = val_dataloader)

"""## Training Visualization
In the following plots, the performance up to epoch 75 is from the previous phase of training. The performance beyond epoch 75 is logged during this fine-tuning stage. The accuracy of the model has improved significantly and the model has also started overfitting. We can also observe the slight impact the learning rate scheduler has made on the logged metrics below.
"""

sns.lineplot(x=range(len(metric_tracker.train_loss)), y=metric_tracker.train_loss, label="train_loss")
sns.lineplot(x=range(len(metric_tracker.val_loss)), y=metric_tracker.val_loss, label="val_loss")
plt.legend()
plt.show()

sns.lineplot(x=range(len(metric_tracker.train_acc)), y=metric_tracker.train_acc, label="train_acc")
sns.lineplot(x=range(len(metric_tracker.val_acc)), y=metric_tracker.val_acc, label="val_acc")
plt.legend()
plt.show()

sns.lineplot(x=range(len(metric_tracker.lr)), y=metric_tracker.lr, label="lr")
plt.legend()
plt.show()

"""## Test
During testing, the best performing (on val_loss) weights are automatically loaded. We can see that accuracy has improved significantly.
"""

trainer.test(model, test_dataloader)

