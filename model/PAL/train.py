import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from multi_directional_encoder import PALEncoder
from conv_attention_decoder import PALDecoder
from discriminator import Discriminator
from classifier import Classifier
from loss import compute_L_D, compute_L_C, compute_L_E
from paired_dataset import PairedDataset

# config
lambda_adv = 0.1
disc_steps = 1
epochs = 10
batch_size = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

# models
encoder = PALEncoder().to(device)
decoder = PALDecoder(d_model=256, vocab_size=30522).to(device)
classifier = Classifier(d_model=256, vocab_size=30522).to(device)
discriminator = Discriminator(d_model=256).to(device)

# optimizers
opt_E = torch.optim.Adam(list(encoder.parameters()) + 
                         list(decoder.parameters()) + 
                         list(classifier.parameters()), lr=1e-4)
opt_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

# dataloader
transform = transforms.Compose([
    transforms.Resize((128, 512)),
    transforms.ToTensor()
])

dataset = PairedDataset("train_list.txt", "images/handwritten", "images/printed", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

for epoch in range(epochs):
    for batch in train_loader:
        x_p, x_h, y_p, y_h = batch  # (B, C, H, W), (B, T) labels
        x_p = x_p.to(device)
        x_h = x_h.to(device)
        y_p = y_p.to(device)
        y_h = y_h.to(device)

        # Encoder-Decoder 업데이트
        encoder.train(); decoder.train(); classifier.train()
        opt_E.zero_grad()

        a_p = encoder(x_p)  # (B, T, D)
        a_h = encoder(x_h)  # (B, T, D)

        loss_E = compute_L_E(a_p, a_h, y_p, y_h, classifier, discriminator, lambda_adv)
        loss_E.backward()
        opt_E.step()

        # Discriminator k-step 업데이트
        for _ in range(disc_steps):
            opt_D.zero_grad()
            a_p_detach = a_p.detach()
            a_h_detach = a_h.detach()
            loss_D = compute_L_D(discriminator, a_p_detach, a_h_detach)
            loss_D.backward()
            opt_D.step()

    print(f"Epoch {epoch} | L_E: {loss_E.item():.4f} | L_D: {loss_D.item():.4f}")