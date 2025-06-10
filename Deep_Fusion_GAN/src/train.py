import os
from typing import List, Tuple

from src.deep_fusion_gan.model import DeepFusionGAN
from src.utils import create_loader, fix_seed


def train() -> Tuple[List[float], List[float], List[float]]:
    fix_seed()

    # Use absolute path to avoid any ambiguity
    data_path = "D:/GAN-codes/Deep_Fusion_GAN/data"
    encoder_weights_path = "D:/GAN-codes/Deep_Fusion_GAN/text_encoder_weights/text_encoder.pth"
    image_save_path = "../gen_images"
    gen_path_save = "../gen_weights"

    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(gen_path_save, exist_ok=True)

    train_loader = create_loader(256, 24, data_path, "train")
    # Create test loader for evaluation metrics
    test_loader = create_loader(256, 24, data_path, "test")
    print(f"Test set size: {len(test_loader.dataset)} images")
    
    model = DeepFusionGAN(n_words=train_loader.dataset.n_words,
                          encoder_weights_path=encoder_weights_path,
                          image_save_path=image_save_path,
                          gen_path_save=gen_path_save)
    
    # Add this line - pass the vocabulary mapping to the model
    model.ixtoword = train_loader.dataset.code2word

    # Explicitly load from the latest checkpoint.pt (after every epoch)
    checkpoint_path = os.path.join(gen_path_save, "checkpoint.pt")
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from latest: {checkpoint_path}")
        start_epoch, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch, is_scores_epoch, fid_scores_epoch, txtimg_losses_epoch = model.load_from_checkpoint(checkpoint_path)
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Starting from scratch.")
        start_epoch = 0
        g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = [], [], []
        is_scores_epoch, fid_scores_epoch, txtimg_losses_epoch = [], [], []

    # Pass loaded metrics, start_epoch, and test_loader to fit
    g_losses, d_losses, d_gp_losses, is_scores, fid_scores, txtimg_losses = model.fit(
        train_loader,
        test_loader=test_loader,  # Pass test loader for metrics
        num_epochs=600,
        start_epoch=start_epoch,
        g_losses_epoch=g_losses_epoch,
        d_losses_epoch=d_losses_epoch,
        d_gp_losses_epoch=d_gp_losses_epoch,
        is_scores_epoch=is_scores_epoch,
        fid_scores_epoch=fid_scores_epoch,
        txtimg_losses_epoch=txtimg_losses_epoch,
        auto_resume=False
    )

    # Print metrics after every epoch
    for epoch in range(len(g_losses)):
        print(f"Epoch {epoch + 1}: "
              f"G Loss: {g_losses[epoch]:.4f}, "
              f"D Loss: {d_losses[epoch]:.4f}, "
              f"D GP Loss: {d_gp_losses[epoch]:.4f}, "
              f"Text-Image Loss: {txtimg_losses[epoch]:.4f}, "
              f"IS: {is_scores[epoch]:.4f}, "
              f"FID: {fid_scores[epoch]:.4f}")

    return g_losses, d_losses, d_gp_losses, is_scores, fid_scores, txtimg_losses


if __name__ == '__main__':
    train()