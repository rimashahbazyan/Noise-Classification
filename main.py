import argparse

from torch.utils.data import DataLoader

from configs import configs
from dataset import NoiseClassesDataset
from model import ConvModel
from model import GRUModel
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="CNN vs GRU", default='CNN')

parser.add_argument("--dataset_dir", type=str, help="Path to dataset audios directory",
                    default='/Users/rshahbazyan/Downloads/alalala/audio/audio/16000')

parser.add_argument("--dataset_csv_path", type=str, help="Path to dataset csv",
                    default='/Users/rshahbazyan/Downloads/alalala/esc50.csv')

args = parser.parse_args()
noise_classes_dataset = NoiseClassesDataset(args.dataset_dir, args.dataset_csv_path)
class_mapping = noise_classes_dataset.get_class_mapping()

train_data_loader = DataLoader(noise_classes_dataset, batch_size=configs['batch_size'])
val_data_loader = DataLoader(noise_classes_dataset, batch_size=8)

if args.model_type == "CNN":
    model = ConvModel(**configs["model_params"]).to(configs["device"])

elif args.model_type == "GRU":
    model = GRUModel(**configs["model_params"]).to(configs["device"])

trainer = Trainer(model=model,
                  log_dir=configs['log_dir'],
                  log_steps=configs['log_steps'],
                  lr=configs['lr'],
                  device=configs["device"],
                  ckpt_path=configs.get("ckpt_path", None))

trainer.train(train_data_loader, val_data_loader)
