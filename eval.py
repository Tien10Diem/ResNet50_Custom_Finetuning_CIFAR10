
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm
from Model.ResNet50 import ResNet50
import torch
from Dataset.dataset import dataset
from torch.utils.data import DataLoader

def test():

    compose = Compose([
        ToTensor(),
        Resize((224,224)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_data = dataset(r"Data\data\cifar10\test", compose)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    progress = tqdm(test_dataloader)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path_best = "best_model.pth"
    model = ResNet50().to(device)
    sta_dict = torch.load(path_best, map_location=device)

    model.load_state_dict(sta_dict['best_model'])

    predict = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(progress):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs= model(imgs)
            predict.extend(torch.argmax(outputs, dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())


    print(accuracy_score(true_labels, predict))
    print(classification_report(true_labels, predict))
    print(confusion_matrix(true_labels, predict))


if __name__ == "__main__":
    test()

