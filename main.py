import argparse

from torch.utils.data import DataLoader

from configs import configs
from dataset import NoiseClassesDataset
from model import ConvModel, HubertDense
from model import GRUModel
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="CNN , GRU or HUB", default='HUB')

parser.add_argument("--dataset_dir", type=str, help="Path to dataset audios directory",
                    default='/home/rshahbazyan/Downloads/UrbanSound8K/audio16k/')

parser.add_argument("--dataset_csv_path", type=str, help="Path to dataset csv",
                    default='/home/rshahbazyan/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv')

args = parser.parse_args()
train_dataset = NoiseClassesDataset(args.dataset_dir+'Train', 
                                    args.dataset_csv_path, 
                                    audio_length=configs["audio_len"],
                                    is_hubert=configs["is_hubert"])
val_dataset = NoiseClassesDataset(args.dataset_dir+'Validation', 
                                  args.dataset_csv_path, 
                                  audio_length=configs["audio_len"],
                                  is_hubert=configs["is_hubert"])
class_mapping = train_dataset.get_class_mapping()

train_data_loader = DataLoader(train_dataset, batch_size=configs['batch_size'])
val_data_loader = DataLoader(val_dataset, batch_size=8)
print("Data Loaded")

if args.model_type == "CNN":
    model = ConvModel(**configs["model_params"]).to(configs["device"])

elif args.model_type == "GRU":
    model = GRUModel(**configs["model_params"]).to(configs["device"])

elif args.model_type == "HUB":
    model = HubertDense(**configs["model_params"]).to(configs["device"])
print("Model Initialized")

trainer = Trainer(model=model,
                  log_dir=configs['log_dir'],
                  log_steps=configs['log_steps'],
                  lr=configs['lr'],
                  device=configs["device"],
                  ckpt_path=configs.get("ckpt_path", None),
                  is_hubert=configs["is_hubert"])

trainer.train(train_data_loader, val_data_loader)
