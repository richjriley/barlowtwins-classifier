import csv
from pathlib import Path

import torch
from torchvision import datasets, transforms
from alive_progress import alive_bar

model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")

image_directory = Path("data", "kagglecatsanddogs_3367a", "PetImages")
num_files = len(list(image_directory.rglob("*.jpeg")))

val_dataset = datasets.ImageFolder(
    image_directory,
    transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ]
    ),
)

val_loader = torch.utils.data.DataLoader(val_dataset)
with open("embeddings.csv", "w") as output_file:
    writer = csv.writer(output_file)

    with alive_bar(num_files) as bar:
        for step, (images, target) in enumerate(val_loader):
            file_name = val_dataset.imgs[step][0]
            output = model(images)
            writer.writerow([*output.data.tolist()[0], file_name, target.data.tolist()[0]])
            bar()
