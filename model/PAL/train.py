import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from multi_directional_encoder import PALEncoder
from conv_attention_decoder import PALDecoder
from discriminator import Discriminator
from classifier import Classifier
from loss import compute_L_D, compute_L_E
from paired_dataset import PairedDataset, load_vocab, make_label_to_ids

# ------------------------------
# 1. Collate: 배치 오른쪽 패딩 + (선택) stride 배수 정렬 + 라벨 패딩
# ------------------------------
def collate_pad_right(batch, multiple_of=32, pad_token_id=0):
    """
    batch: list of (x_p, x_h, y_p, y_h)
      x_*: [1,H,W_i]  (0~1, 이미 ToTensor/Normalize 적용)
      y_*: [L_i]      (이미 sos/eos 포함, Dataset에서 label_to_ids로 변환됨)
    returns:
      x_p, x_h: [B,1,H,Wmax]
      mask_p, mask_h: [B,1,H,Wmax]  (유효영역=1, 패딩=0)
      y_p, y_h: [B,Lmax]            (pad_token_id로 패딩)
      len_p, len_h: [B]
    """
    xs_p, xs_h, ys_p, ys_h = [], [], [], []
    for x_p, x_h, y_p, y_h in batch:
        xs_p.append(x_p); xs_h.append(x_h)
        ys_p.append(y_p); ys_h.append(y_h)

    H = xs_p[0].shape[-2]
    widths = [t.shape[-1] for t in xs_p] + [t.shape[-1] for t in xs_h]
    Wmax = max(widths)
    if multiple_of and multiple_of > 1:
        Wmax = ((Wmax + multiple_of - 1) // multiple_of) * multiple_of  # stride 정렬

    B = len(batch)
    dtype = xs_p[0].dtype
    x_p_batch = torch.ones((B,1,H,Wmax), dtype=dtype)   # 흰 배경=1.0
    x_h_batch = torch.ones((B,1,H,Wmax), dtype=dtype)
    mask_p = torch.zeros((B,1,H,Wmax), dtype=torch.bool)
    mask_h = torch.zeros((B,1,H,Wmax), dtype=torch.bool)

    for i,(xp,xh) in enumerate(zip(xs_p,xs_h)):
        wp, wh = xp.shape[-1], xh.shape[-1]
        x_p_batch[i, :, :, :wp] = xp;  mask_p[i, :, :, :wp] = True
        x_h_batch[i, :, :, :wh] = xh;  mask_h[i, :, :, :wh] = True

    len_p = torch.tensor([len(y) for y in ys_p], dtype=torch.long)
    len_h = torch.tensor([len(y) for y in ys_h], dtype=torch.long)
    Lp, Lh = int(len_p.max()), int(len_h.max())
    y_p_batch = torch.full((B,Lp), pad_token_id, dtype=torch.long)
    y_h_batch = torch.full((B,Lh), pad_token_id, dtype=torch.long)
    for i,(yp,yh) in enumerate(zip(ys_p,ys_h)):
        y_p_batch[i,:len(yp)] = yp
        y_h_batch[i,:len(yh)] = yh

    return x_p_batch, x_h_batch, mask_p, mask_h, y_p_batch, y_h_batch, len_p, len_h

def main():
    # ------------------------------
    # Config
    # ------------------------------
    lambda_adv = 0.1
    disc_steps = 1
    epochs = 10
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Vocab 로드 -> label_to_ids 생성
    base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
    vocab_path = os.path.normpath(os.path.join(base_dir, "crohme_vocab.txt"))
    token2id, pad_id, sos_id, eos_id, unk_id = load_vocab(vocab_path)
    label_to_ids = make_label_to_ids(
        token2id, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id,
        add_sos=True, add_eos=True
    )

    # ------------------------------
    # Dataset / DataLoader
    # ------------------------------
    list_path = os.path.normpath(os.path.join(base_dir, "dummy/caption.txt"))
    img_dir_h = os.path.normpath(os.path.join(base_dir, "dummy/hme_preprocessed"))
    img_dir_p = os.path.normpath(os.path.join(base_dir, "dummy/pme_preprocessed"))

    transform = T.Compose([T.ToTensor(),T.Normalize((0.5,), (0.5,))])

    dataset = PairedDataset(
        list_path, img_dir_h, img_dir_p,
        transform=transform,
        label_to_ids=label_to_ids,
    )

    num_workers = 0 if os.name == "nt" else 4
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate_pad_right(b, multiple_of=32, pad_token_id=pad_id),
    )

    # ------------------------------
    # Models (vocab_size는 vocab에 맞춰 설정)
    # ------------------------------
    vocab_size = len(token2id)
    encoder = PALEncoder().to(device)
    decoder = PALDecoder(d_model=256, vocab_size=vocab_size).to(device)
    classifier = Classifier(d_model=256, vocab_size=vocab_size).to(device)
    discriminator = Discriminator(d_model=256).to(device)

    # ------------------------------
    # Optimizers
    # ------------------------------
    opt_E = torch.optim.Adam(list(encoder.parameters()) + 
                            list(decoder.parameters()) + 
                            list(classifier.parameters()), lr=1e-4)
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # ------------------------------
    # Train loop
    # ------------------------------
    for epoch in range(epochs):
        encoder.train(); decoder.train(); classifier.train(); discriminator.train()
        for batch in train_loader:
            x_p, x_h, mask_p, mask_h, y_p, y_h, len_p, len_h = [t.to(device) if torch.is_tensor(t) else t for t in batch]

            # ---- Encoder/Decoder/Classifier step ----
            opt_E.zero_grad()
            a_p = encoder(x_p)   # (B, T, D)
            a_h = encoder(x_h)

            # compute_L_E가 마스크/길이를 받도록 설계된 경우, 여기에 mask/len 전달하도록 수정
            loss_E = compute_L_E(a_p, a_h, y_p, y_h, classifier, discriminator, lambda_adv)
            loss_E.backward()
            opt_E.step()

            # ---- Discriminator k steps ----
            for _ in range(disc_steps):
                opt_D.zero_grad()
                loss_D = compute_L_D(discriminator, a_p.detach(), a_h.detach())
                loss_D.backward()
                opt_D.step()

        print(f"Epoch {epoch} | L_E: {loss_E.item():.4f} | L_D: {loss_D.item():.4f}")
    
if __name__ == "__main__":
    main()