import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DreamBoothDataset(Dataset):
    def __init__(self, folder_path, token="sks_person"):
        self.folder_path = folder_path
        self.image_paths = [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # Уменьшаем размер изображений до 128×128 для снижения потребления памяти
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.token = token

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        caption = f"a photo of {self.token}"
        return {"pixel_values": image, "caption": caption}
