import os

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class Trainer:
    def __init__(self, model: nn.Module, log_steps, log_dir, lr, device, ckpt_path=None, is_hubert=False):
        self.log_dir = log_dir
        self.log_step = log_steps
        self.device = device
        self.step = 0

        self.model = model.to(self.device)
        self.log_writer = SummaryWriter(logdir=os.path.join(log_dir, self.model.__name__, "tensorboard"))
        self.loss = CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.hubert_model = None
        if is_hubert:
            self.hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(self.device)

        if ckpt_path is not None:
            self.load_ckpt(ckpt_path)

    def train(self, train_dataloader, val_dataloader):
        while True:
            for data, label in train_dataloader:
                with torch.set_grad_enabled(True):
                    self.model.train()
                    data = data.to(self.device)
                    if self.hubert_model:
                        with torch.no_grad():
                            data = self.hubert_model(data).last_hidden_state
                    y_pred = self.model(data)
                    label = label.to(self.device)
                    train_loss = self.loss(y_pred, label)
                    self.optimizer.zero_grad()
                    train_loss.backward()
                    self.optimizer.step()
                    print(f"STEP:{self.step}    LOSS:{train_loss}")
                    self.step += 1
                if self.step % self.log_step == self.log_step - 1:
                    self.model.eval()
                    val_data, val_label = next(iter(val_dataloader))
                    val_data = val_data.to(self.device)
                    if self.hubert_model:
                        with torch.no_grad():
                            val_data = self.hubert_model(val_data).last_hidden_state
                    y_val_pred = self.model(val_data)
                    val_label = val_label.to(self.device)
                    val_loss = self.loss(y_val_pred, val_label)
                    self.log_loss(train_loss, val_loss)
                    self.save_ckpt()

    def log_loss(self, train_loss, val_loss):
        self.log_writer.add_scalar("TRAIN_LOSS", train_loss, self.step)
        self.log_writer.add_scalar("VAL_LOSS", val_loss, self.step)

    def save_ckpt(self):
        dir = os.path.join(self.log_dir, self.model.__name__, "checkpoints")
        checkpoint = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step':                 self.step
        }
        os.makedirs(dir, exist_ok=True)

        torch.save(checkpoint, os.path.join(dir, f"{self.step}.pt"))

    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
