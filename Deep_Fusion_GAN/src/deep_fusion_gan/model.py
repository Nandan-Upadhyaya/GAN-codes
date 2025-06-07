import os.path
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import trange

from src.discriminator.model import Discriminator
from src.generator.model import Generator
from src.objects.utils import prepare_data
from src.text_encoder.model import RNNEncoder
from src.damsm.loss import damsm_loss
from src.evaluation.metrics import compute_inception_score, compute_fid  # You must provide these


class DeepFusionGAN:
    def __init__(self, n_words, encoder_weights_path: str, image_save_path: str, gen_path_save: str):
        super().__init__()
        self.image_save_path = image_save_path
        self.gen_path_save = gen_path_save

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(n_channels=32, latent_dim=100).to(self.device)
        self.discriminator = Discriminator(n_c=32).to(self.device)

        self.text_encoder = RNNEncoder.load(encoder_weights_path, n_words)
        self.text_encoder.to(self.device)

        # Add this line - get the word mapping from the dataset
        self.ixtoword = None  # Will be set from the outside

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # Reduce learning rates for stability
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))  # Reduced from 0.0001
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))  # Reduced from 0.0004

        self.relu = nn.ReLU()

        # Mixed precision training
        # self.scaler_g = torch.cuda.amp.GradScaler()
        # self.scaler_d = torch.cuda.amp.GradScaler()
        
        # Add gradient clipping thresholds for stability
        self.grad_clip_g = 5.0
        self.grad_clip_d = 5.0

    def _zero_grad(self):
        self.d_optim.zero_grad()
        self.g_optim.zero_grad()

    def _compute_gp(self, images: Tensor, sentence_embeds: Tensor) -> Tensor:
        batch_size = images.shape[0]

        images_interpolated = images.data.requires_grad_()
        sentences_interpolated = sentence_embeds.data.requires_grad_()

        embeds = self.discriminator.build_embeds(images_interpolated)
        logits = self.discriminator.get_logits(embeds, sentences_interpolated)

        grad_outputs = torch.ones_like(logits)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=(images_interpolated, sentences_interpolated),
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True
        )

        grad_0 = grads[0].reshape(batch_size, -1)
        grad_1 = grads[1].reshape(batch_size, -1)

        grad = torch.cat((grad_0, grad_1), dim=1)
        # Use more stable gradient norm computation with small epsilon
        grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1e-8)
        
        # Clamp gradient norms for stability
        return torch.clamp(grad_norm, 0, 10)

    def compute_text_image_loss(self, fake_images, captions, cap_lens):
        # Call the DAMSM loss function
        return damsm_loss(fake_images, captions, cap_lens, self.text_encoder)

    def compute_is_fid(self, fake_imgs, real_imgs):
        # Ensure images are float32 for Inception model (avoid HalfTensor/FloatTensor mismatch)
        fake_imgs = fake_imgs.float()
        real_imgs = real_imgs.float()
        is_score = compute_inception_score(fake_imgs, cuda=True, batch_size=8, splits=1)
        fid_score = compute_fid(real_imgs, fake_imgs, cuda=True, batch_size=8)
        return is_score, fid_score

    def _check_nan(self, tensor_value, name=""):
        """Check for NaNs and return zero tensor if found to prevent divergence"""
        if torch.isnan(tensor_value).any() or torch.isinf(tensor_value).any():
            print(f"Warning: {name} contains NaN or Inf values, resetting to small random values")
            return torch.randn_like(tensor_value) * 0.01
        return tensor_value

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from a specific checkpoint file"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        
        # Load model weights
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        
        # Load optimizer states
        self.g_optim.load_state_dict(checkpoint['g_optimizer'])
        self.d_optim.load_state_dict(checkpoint['d_optimizer'])
        
        # Store metrics history
        g_losses = checkpoint.get('g_losses', [])
        d_losses = checkpoint.get('d_losses', [])
        d_gp_losses = checkpoint.get('d_gp_losses', [])
        is_scores = checkpoint.get('is_scores', [])
        fid_scores = checkpoint.get('fid_scores', [])
        txtimg_losses = checkpoint.get('txtimg_losses', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}.")
        print(f"Metrics at checkpoint - FID: {fid_scores[-1]:.4f}, IS: {is_scores[-1]:.4f}")
        
        # Return the data needed for continuing training
        return checkpoint['epoch'], g_losses, d_losses, d_gp_losses, is_scores, fid_scores, txtimg_losses

    def fit(self, train_loader: DataLoader, num_epochs: int = 600, start_epoch: int = 0,
            g_losses_epoch=None, d_losses_epoch=None, d_gp_losses_epoch=None,
            is_scores_epoch=None, fid_scores_epoch=None, txtimg_losses_epoch=None,
            auto_resume: bool = True):
        # Use provided histories or initialize
        g_losses_epoch = g_losses_epoch if g_losses_epoch is not None else []
        d_losses_epoch = d_losses_epoch if d_losses_epoch is not None else []
        d_gp_losses_epoch = d_gp_losses_epoch if d_gp_losses_epoch is not None else []
        is_scores_epoch = is_scores_epoch if is_scores_epoch is not None else []
        fid_scores_epoch = fid_scores_epoch if fid_scores_epoch is not None else []
        txtimg_losses_epoch = txtimg_losses_epoch if txtimg_losses_epoch is not None else []

        # Disable internal auto-resume if requested
        if auto_resume and start_epoch == 0 and os.path.exists(os.path.join(self.gen_path_save, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(self.gen_path_save, "checkpoint.pt"))
            start_epoch = checkpoint['epoch'] + 1
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.g_optim.load_state_dict(checkpoint['g_optimizer'])
            self.d_optim.load_state_dict(checkpoint['d_optimizer'])
            g_losses_epoch = checkpoint['g_losses']
            d_losses_epoch = checkpoint['d_losses']
            d_gp_losses_epoch = checkpoint['d_gp_losses']
            is_scores_epoch = checkpoint['is_scores']
            fid_scores_epoch = checkpoint['fid_scores']
            txtimg_losses_epoch = checkpoint['txtimg_losses']
            print(f"Resuming from epoch {start_epoch}")

        for epoch in trange(start_epoch, num_epochs, desc="Train Deep Fusion GAN"):
            g_losses, d_losses, d_gp_losses, txtimg_losses = [], [], [], []
            for batch in train_loader:
                images, captions, captions_len, _ = prepare_data(batch, self.device)
                batch_size = images.shape[0]

                # Remove autocast context
                # with torch.cuda.amp.autocast():
                sentence_embeds = self.text_encoder(captions, captions_len).detach()

                real_embeds = self.discriminator.build_embeds(images)
                real_logits = self.discriminator.get_logits(real_embeds, sentence_embeds)
                d_loss_real = self.relu(1.0 - real_logits).mean()

                shift_embeds = real_embeds[:(batch_size - 1)]
                shift_sentence_embeds = sentence_embeds[1:batch_size]
                shift_real_image_embeds = self.discriminator.get_logits(shift_embeds, shift_sentence_embeds)
                d_loss_mismatch = self.relu(1.0 + shift_real_image_embeds).mean()

                noise = torch.randn(batch_size, 100).to(self.device)
                fake_images = self.generator(noise, sentence_embeds)

                fake_embeds = self.discriminator.build_embeds(fake_images.detach())
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)
                d_loss_fake = self.relu(1.0 + fake_logits).mean()

                d_loss = d_loss_real + (d_loss_fake + d_loss_mismatch) / 2.0
                d_loss = self._check_nan(d_loss, "discriminator loss")

                self._zero_grad()
                d_loss.backward()
                # Add gradient clipping for discriminator
                if self.grad_clip_d > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_d)
                self.d_optim.step()

                d_losses.append(d_loss.item())

                # with torch.cuda.amp.autocast():
                grad_l2norm = self._compute_gp(images, sentence_embeds)
                d_loss_gp = 2.0 * torch.mean(grad_l2norm ** 6)
                d_loss_gp = self._check_nan(d_loss_gp, "gradient penalty loss")

                self._zero_grad()
                d_loss_gp.backward()
                if self.grad_clip_d > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_d)
                self.d_optim.step()

                d_gp_losses.append(d_loss_gp.item())

                # with torch.cuda.amp.autocast():
                fake_embeds = self.discriminator.build_embeds(fake_images)
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)
                g_loss = -fake_logits.mean()
                g_loss = self._check_nan(g_loss, "generator loss")
                txtimg_loss = 0.0
                if hasattr(self, "compute_text_image_loss"):
                    txtimg_loss = self.compute_text_image_loss(fake_images, captions, captions_len)
                    txtimg_loss = self._check_nan(txtimg_loss, "text-image loss")

                self._zero_grad()
                g_loss.backward()
                if self.grad_clip_g > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_g)
                self.g_optim.step()

                g_losses.append(g_loss.item())
                txtimg_losses.append(txtimg_loss if isinstance(txtimg_loss, float) else float(txtimg_loss))

            g_losses_epoch.append(np.mean(g_losses))
            d_losses_epoch.append(np.mean(d_losses))
            d_gp_losses_epoch.append(np.mean(d_gp_losses))
            txtimg_losses_epoch.append(np.mean(txtimg_losses))

            # Compute IS and FID for validation set (or train set if no val loader)
            with torch.no_grad():
                # Collect real and fake images for metrics (sample up to 512 for speed)
                real_imgs_metric, fake_imgs_metric = [], []
                for i, batch in enumerate(train_loader):
                    if i >= 32: break  # 32*16=512 images if batch size 16
                    images, captions, captions_len, _ = prepare_data(batch, self.device)
                    sentence_embeds = self.text_encoder(captions, captions_len).detach()
                    noise = torch.randn(images.size(0), 100).to(self.device)
                    with torch.cuda.amp.autocast():
                        fake_images = self.generator(noise, sentence_embeds)
                    real_imgs_metric.append(images.cpu())
                    fake_imgs_metric.append(fake_images.cpu())
                real_imgs_metric = torch.cat(real_imgs_metric, dim=0)
                fake_imgs_metric = torch.cat(fake_imgs_metric, dim=0)
                # Replace with your actual IS/FID computation functions
                is_score, fid_score = 0.0, 0.0
                if hasattr(self, "compute_is_fid"):
                    is_score, fid_score = self.compute_is_fid(fake_imgs_metric, real_imgs_metric)
                is_scores_epoch.append(is_score)
                fid_scores_epoch.append(fid_score)

            # Save model and samples after every epoch
            self._save_fake_image_with_prompt(fake_images, captions, captions_len, epoch)
            self._save_checkpoint(epoch, g_losses_epoch, d_losses_epoch, d_gp_losses_epoch, 
                                is_scores_epoch, fid_scores_epoch, txtimg_losses_epoch)

            # Print epoch metrics
            print(f"Epoch {epoch+1}: "
                  f"G Loss: {g_losses_epoch[-1]:.4f}, "
                  f"D Loss: {d_losses_epoch[-1]:.4f}, "
                  f"D GP Loss: {d_gp_losses_epoch[-1]:.4f}, "
                  f"Text-Image Loss: {txtimg_losses_epoch[-1]:.4f}, "
                  f"IS: {is_scores_epoch[-1]:.4f}, "
                  f"FID: {fid_scores_epoch[-1]:.4f}")

        return g_losses_epoch, d_losses_epoch, d_gp_losses_epoch, is_scores_epoch, fid_scores_epoch, txtimg_losses_epoch

    def _save_checkpoint(self, epoch, g_losses, d_losses, d_gp_losses, is_scores, fid_scores, txtimg_losses):
        """Save complete checkpoint for training resumption"""
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optim.state_dict(),
            'd_optimizer': self.d_optim.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses,
            'd_gp_losses': d_gp_losses,
            'is_scores': is_scores,
            'fid_scores': fid_scores,
            'txtimg_losses': txtimg_losses
        }
        
        # Save latest checkpoint for resumption
        torch.save(checkpoint, os.path.join(self.gen_path_save, "checkpoint.pt"))
        
        # Also save numbered checkpoint every 10 epochs for safety
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(self.gen_path_save, f"checkpoint_epoch_{epoch}.pt"))
        
        # Save generator weights separately (as before)
        self._save_gen_weights(epoch)
    
    def _save_gen_weights(self, epoch: int):
        """Save only the generator weights (useful for inference)"""
        gen_path = os.path.join(self.gen_path_save, f"gen_{epoch}.pth")
        torch.save(self.generator.state_dict(), gen_path)
            
    def _save_fake_image_with_prompt(self, fake_images: Tensor, captions, cap_lens, epoch: int):
        """Save images with their corresponding text prompts"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            # Select a subset of images to visualize (at most 4)
            num_images = min(4, fake_images.size(0))
            
            # Create figure with subplots
            fig = Figure(figsize=(12, 3*num_images))
            canvas = FigureCanvasAgg(fig)
            
            # Get original text from captions
            original_text = []
            for i in range(num_images):
                sent = ""
                for j in range(cap_lens[i]):
                    word_idx = captions[i][j].item()
                    # Use self.ixtoword instead of self.text_encoder.ixtoword
                    if self.ixtoword is not None and word_idx in self.ixtoword:
                        sent += self.ixtoword[word_idx] + " "
                    else:
                        sent += f"[{word_idx}] "
                original_text.append(sent.strip())
                
            # For each image and caption
            for i in range(num_images):
                # Get image and convert to numpy array for plotting
                img = fake_images[i].detach().cpu().float()
                img = (img + 1) / 2  # [-1,1] to [0,1]
                img = img.numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                # Add subplot
                ax = fig.add_subplot(num_images, 1, i+1)
                ax.imshow(img)
                ax.set_title(f"Prompt: {original_text[i]}", fontsize=10)
                ax.axis('off')
                
            plt.tight_layout()
            
            # Save the figure
            img_path = os.path.join(self.image_save_path, f"samples_with_text_epoch_{epoch}.jpg")
            fig.savefig(img_path)
            
            # Also save the regular grid of images (as before)
            self._save_fake_image(fake_images, epoch)
            
        except Exception as e:
            print(f"Error saving images with text: {e}")
            # Fallback to regular image saving
            self._save_fake_image(fake_images, epoch)
    
    def _save_fake_image(self, fake_images: Tensor, epoch: int):
        """Save a grid of generated images"""
        img_path = os.path.join(self.image_save_path, f"fake_sample_epoch_{epoch}.png")
        vutils.save_image(fake_images.data, img_path, normalize=True)