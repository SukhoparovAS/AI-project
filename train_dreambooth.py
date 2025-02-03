import os
# Устанавливаем переменную окружения для управления распределением памяти
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from dreambooth_dataset import DreamBoothDataset
from PIL import Image
import numpy as np

# Устройство: используем GPU, если доступен
device = "cuda" if torch.cuda.is_available() else "cpu"

# Идентификатор базовой модели Stable Diffusion
model_id = "runwayml/stable-diffusion-v1-5"
latent_scale = 0.18215  # Масштаб для латентного пространства

print("Загрузка компонентов модели...")

# Загружаем VAE в FP16 (замораживаем)
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16).to(device)
vae.requires_grad_(False)

# Загружаем токенизатор и текстовый энкодер в FP16 (замораживаем)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
text_encoder.requires_grad_(False)

# Загружаем UNet в FP32 для обучения (GradScaler требует мастер-параметры в FP32)
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32).to(device)
unet.enable_gradient_checkpointing()  # включаем gradient checkpointing для экономии памяти

# Загружаем scheduler для диффузионного процесса
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Создаем датасет и DataLoader (учтите, что изображения теперь 128×128)
dataset = DreamBoothDataset(folder_path="./subject_images", token="sks_person")
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Оптимизатор
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-6)

# GradScaler для AMP
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

# Параметры обучения
num_train_steps = 500
accumulation_steps = 1  # обновление весов каждые 2 шага
step_count = 0

unet.train()
print("Начало обучения...")

for step in range(num_train_steps):
    for batch in dataloader:
        if step_count % accumulation_steps == 0:
            optimizer.zero_grad()

        # Переносим изображения на устройство и переводим в FP16
        pixel_values = batch["pixel_values"].to(device).half()
        captions = batch["caption"]

        # Токенизация текстовых подписей
        inputs = tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(device)

        # Получаем текстовые эмбеддинги без градиентов
        with torch.no_grad():
            text_embeddings = text_encoder(input_ids)[0]

        # Выполняем forward-проход в режиме AMP (FP16)
        with torch.autocast("cuda", dtype=torch.float16):
            # Кодирование изображений в латентное пространство через VAE
            latents = vae.encode(pixel_values).latent_dist.sample() * latent_scale

            # Сохраняем отладочное изображение каждые 50 шагов
            if (step_count + 1) % 250 == 0:
                with torch.no_grad():
                    decoded = vae.decode(latents / latent_scale).sample
                decoded_img = (decoded.clamp(-1, 1) + 1) / 2  # нормировка в диапазон [0,1]
                decoded_img = decoded_img.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
                Image.fromarray(decoded_img.astype(np.uint8)).save(f"debug_decoded_step_{step_count+1}.jpg")
                print(f"[Step {step_count+1}] Сохранено отладочное изображение: debug_decoded_step_{step_count+1}.jpg")

            # Выбираем случайные временные шаги
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()

            # Генерируем случайный шум (FP16)
            noise = torch.randn(latents.shape, device=device, dtype=torch.float16)

            # Добавляем шум к латентам согласно расписанию scheduler-а
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Прогон через UNet для предсказания шума
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample

            # Вычисляем MSE потери между предсказанным шумом и истинным шумом
            loss = nn.MSELoss()(noise_pred, noise)

        scaler.scale(loss).backward()
        step_count += 1

        if step_count % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            print(f"Step {step_count} - Loss: {loss.item():.6f}")

        torch.cuda.empty_cache()

        # Обрабатываем один батч за итерацию внешнего цикла
        break

# Сохраняем дообученную модель
os.makedirs("./dreambooth_model", exist_ok=True)
unet.save_pretrained("./dreambooth_model")
print("Обучение завершено, модель сохранена в ./dreambooth_model")
