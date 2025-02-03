import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

print("Загрузка компонентов модели для генерации...")

# Загружаем компоненты базовой модели
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Загружаем дообученный UNet (для инференса можно использовать FP16)
unet = UNet2DConditionModel.from_pretrained("./dreambooth_model", torch_dtype=torch.float16).to(device)

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,  # Отключаем safety_checker для отладки
    feature_extractor=None
)
pipe = pipe.to(device)

prompt = "sks_person, a photorealistic portrait of a person in high-quality studio lighting."
print("Запрос:", prompt)

with torch.autocast("cuda", dtype=torch.float16):
    result = pipe(prompt, num_inference_steps=100, guidance_scale=8)

image = result.images[0]
image_np = np.array(image).astype(np.float32)
print("Статистика сгенерированного изображения:")
print(f"Min: {image_np.min()}, Max: {image_np.max()}, Mean: {image_np.mean()}")
image.save("output.jpg")
print("Изображение сохранено как output.jpg")
