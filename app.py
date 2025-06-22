import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from model import ConvAutoencoder

# Configura dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para carregar o modelo e os pesos
@st.cache_resource
def load_model():
    model = ConvAutoencoder().to(device)
    model.load_state_dict(torch.load("conv_autoencoder_mnist.pth", map_location=device))
    model.eval()
    return model

# Função para gerar imagens aproximadas do dígito dado
def generate_images(model, digit, n=5):
    # Carrega dataset MNIST para obter imagens do dígito
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    latents = []
    with torch.no_grad():
        for imgs, labels in loader:
            mask = (labels == digit)
            if mask.sum() == 0:
                continue
            imgs_digit = imgs[mask].to(device)
            z = model.encode(imgs_digit)
            latents.append(z)
    latents = torch.cat(latents, dim=0)

    mean_latent = latents.mean(dim=0, keepdim=True)  # [1,64,1,1]

    imgs_generated = []
    for _ in range(n):
        noise = torch.randn_like(mean_latent) * 0.1
        sample_latent = mean_latent + noise
        img = model.decode(sample_latent).cpu()
        imgs_generated.append(img.squeeze().numpy())
    return imgs_generated

# Interface Streamlit
st.title("Gerador de Dígitos Manuscritos")

digit = st.number_input("Digite um número entre 0 e 9:", min_value=0, max_value=9, step=1)
if st.button("Gerar 5 imagens"):
    with st.spinner("Carregando modelo e gerando imagens..."):
        model = load_model()
        images = generate_images(model, digit, n=5)

    cols = st.columns(5)
    for i, img in enumerate(images):
        cols[i].image(img, width=100)