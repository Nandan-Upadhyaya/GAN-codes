import torch
import torch.nn.functional as F

def damsm_loss(fake_images, captions, cap_lens, text_encoder):
    # fake_images: (B, 3, H, W)
    # captions: (B, max_len, 1)
    # cap_lens: (B,)
    # text_encoder: RNNEncoder
    # This is a minimal version: use global average pooling on images and cosine similarity to sentence embedding

    # Get image features (global average pooling)
    img_feat = torch.mean(fake_images, dim=[2, 3])  # (B, 3)
    # Get sentence embeddings
    sent_emb = text_encoder(captions, cap_lens)  # (B, embed_dim)
    # Project image features to same dim as sent_emb if needed
    if img_feat.size(1) != sent_emb.size(1):
        img_feat = F.linear(img_feat, torch.eye(sent_emb.size(1), img_feat.size(1), device=img_feat.device))
    # Normalize
    img_feat = F.normalize(img_feat, dim=1)
    sent_emb = F.normalize(sent_emb, dim=1)
    # Cosine similarity
    scores = torch.sum(img_feat * sent_emb, dim=1)
    # DAMSM loss: negative mean cosine similarity
    loss = 1 - scores.mean()
    return loss
