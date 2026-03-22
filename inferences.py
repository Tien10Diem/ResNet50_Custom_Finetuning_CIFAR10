from Model.ResNet50 import ResNet50
import torch
import cv2 as cv
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
import os
import argparse as parse


def args():
    arg = parse.ArgumentParser()
    arg.add_argument('--image', '-i', default = "R.jpg") 
    arg.add_argument('--weight', '-w', default = "best_model.pth") 
    return arg.parse_args()


def pred(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    compose = Compose([
    ToTensor(),
    Resize((224,224)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = compose(img)
    img = img.unsqueeze(0)
    
    model = ResNet50()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(args.weight, map_location=device)['best_model'])
    model.eval()

    with torch.no_grad():
        output = model(img)

    pred = torch.argmax(output, dim=1).to("cpu")
    mapping = sorted(os.listdir(r"Data\data\cifar10\train"))
    return mapping[pred]

if __name__ == "__main__":
    
    args = args()
    predict = pred(args.image)
    print(predict)
    

