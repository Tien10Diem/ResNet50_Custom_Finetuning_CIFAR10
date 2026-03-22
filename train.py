from sklearn.metrics import accuracy_score
from torchvision.models import ResNet50_Weights
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from Model.ResNet50 import ResNet50
import torch
from Dataset.dataset import dataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn


weights = ResNet50_Weights.IMAGENET1K_V2

def load_model():
  model = ResNet50()

  mydict= model.state_dict()
  office= weights.get_state_dict(check_hash=True)

  office.pop('fc.weight', None)
  office.pop('fc.bias', None)

  list_key_r50= list(office.keys())
  list_key_mymodel = list(mydict.keys())

  new_dict ={}
  for i in range(len(list_key_r50)):
    if office[list_key_r50[i]].shape != mydict[list_key_mymodel[i]].shape:
      print(f"shape khác nhau ở: {list_key_r50[i]} và {list_key_mymodel[i]}")

    else:
      new_dict[list_key_mymodel[i]] = office[list_key_r50[i]]

  model.load_state_dict(new_dict, strict=False)
  return model


def train(resume = None, epochs = 3):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Using {device} device")

  compose = Compose([
      ToTensor(),
      Resize((224,224)),
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  training_data = dataset(r"Data\data\cifar10\train", compose)

  generator = torch.Generator().manual_seed(42)
  train_ds, val_ds = random_split(training_data, [0.9, 0.1], generator=generator)

  train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=False)

  model = load_model().to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
  criterion = nn.CrossEntropyLoss()

  if resume is not None:
    sta_dict = torch.load(resume, map_location=device) if isinstance(resume, str) else resume

    epoch = sta_dict['epoch']
    model.load_state_dict(sta_dict['last_model'])
    optimizer.load_state_dict(sta_dict['optimizer'])
    best_loss = sta_dict['best_loss']
    best_model = sta_dict['best_model']

  else:
    sta_dict = {}
    epoch = 0
    best_loss = 10000000


  for epoch in range(epoch, epochs):
    print(f"Epoch: {epoch+1}")
    total_loss = 0

    progress = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}")
    model.train()
    for idx, (imgs, labels) in (enumerate(progress)):
      imgs = imgs.to(device)
      labels = labels.to(device)

      outputs= model(imgs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      progress.set_postfix(loss=loss.item())
    print(f"Loss train: {total_loss/len(train_dataloader)}")

    model.eval()
    total_loss = 0
    all_labels = []
    all_predict = []
    with torch.no_grad():
      for idx, (imgs, labels) in enumerate(val_dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        all_labels.extend(labels.cpu().tolist())
        outputs= model(imgs)
        all_predict.extend(torch.argmax(outputs, dim=1).cpu().tolist())
        loss = criterion(outputs, labels)

        total_loss += loss.item()

    print(f"Loss val: {total_loss/len(val_dataloader)}")
    print(accuracy_score(all_labels, all_predict))

    if total_loss/len(val_dataloader) < best_loss:
      best_loss = total_loss/len(val_dataloader)
      sta_dict["best_loss"] = total_loss/len(val_dataloader)
      sta_dict["best_model"] = model.state_dict()
      sta_dict['best_opimizer'] = optimizer.state_dict()
      sta_dict['best_epoch'] = epoch
      torch.save(sta_dict, "best_model.pth")
      print("Save best model")

    sta_dict["last_model"] = model.state_dict()
    sta_dict['optimizer'] = optimizer.state_dict()
    sta_dict['epoch'] = epoch
    torch.save(sta_dict, "last_model.pth")
    print("Save last model")


if __name__ == "__main__":
    train()