import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from paired_dataset import PairedDataset, load_vocab, make_label_to_ids
from multi_directional_encoder import PALEncoder
from conv_attention_decoder import PALDecoder
from discriminator import Discriminator
from classifier import Classifier
from loss import compute_L_D
from train import collate_pad_right

def smoke_test(train_loader, encoder, decoder, classifier, discriminator, device, lambda_adv=0.1):
    batch = next(iter(train_loader))
    x_p, x_h, mask_p, mask_h, y_p, y_h, len_p, len_h = batch
    x_p = x_p.to(device); x_h = x_h.to(device)
    y_p = y_p.to(device); y_h = y_h.to(device)

    print("[shapes]")
    print("x_p:", x_p.shape, "x_h:", x_h.shape)
    print("y_p:", y_p.shape, "y_h:", y_h.shape)

    encoder.train(); decoder.train(); classifier.train(); discriminator.train()
    opt_E = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + list(classifier.parameters()),
        lr=1e-4
    )
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    # E/C step
    opt_E.zero_grad()
    a_p = encoder(x_p)   # 기대: [B, T, D]
    a_h = encoder(x_h)
    print("a_p:", a_p.shape, "a_h:", a_h.shape)

    # 에러 때문에 임시로 loss_E 대체 사용
    # 평탄화 이슈 해결하기!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    loss_E = torch.nn.functional.mse_loss(a_p, a_h)
    assert torch.isfinite(loss_E), "loss_E is NaN/Inf"
    loss_E.backward()
    opt_E.step()

    # D step
    opt_D.zero_grad()
    loss_D = compute_L_D(discriminator, a_p.detach(), a_h.detach())
    assert torch.isfinite(loss_D), "loss_D is NaN/Inf"
    loss_D.backward()
    opt_D.step()

    print(f"[smoke] L_E={loss_E.item():.4f} | L_D={loss_D.item():.4f}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = "C:/Users/sophi/OneDrive/바탕 화면/Folder/2025 HYU/인공지능프로젝트/data/crohme"
    vocab_path = os.path.normpath(os.path.join(base_dir, "crohme_vocab.txt"))
    list_path  = os.path.normpath(os.path.join(base_dir, "dummy/caption.txt"))
    img_dir_h  = os.path.normpath(os.path.join(base_dir, "dummy/hme_preprocessed"))
    img_dir_p  = os.path.normpath(os.path.join(base_dir, "dummy/pme_preprocessed"))

    token2id, pad_id, sos_id, eos_id, unk_id = load_vocab(vocab_path)
    label_to_ids = make_label_to_ids(token2id, sos_id, eos_id, unk_id, add_sos=True, add_eos=True)

    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])

    dataset = PairedDataset(
        list_path, img_dir_h, img_dir_p,
        transform=transform,
        label_to_ids=label_to_ids
    )

    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,              # Windows면 0부터!
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate_pad_right(b, multiple_of=32, pad_token_id=pad_id),
    )

    vocab_size = len(token2id)
    encoder = PALEncoder().to(device)
    decoder = PALDecoder(d_model=256, vocab_size=vocab_size).to(device)
    classifier = Classifier(d_model=256, vocab_size=vocab_size).to(device)
    discriminator = Discriminator(d_model=256).to(device)

    smoke_test(train_loader, encoder, decoder, classifier, discriminator, device, lambda_adv=0.1)