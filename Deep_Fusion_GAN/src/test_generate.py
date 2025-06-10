import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import transforms

from src.deep_fusion_gan.model import DeepFusionGAN
from src.text_encoder.model import RNNEncoder
from src.generator.model import Generator

def preprocess_caption(caption, word2code, max_len=18):
    # Tokenize and encode the caption
    tokens = caption.lower().replace('.', '').replace(',', '').split()
    encoded = [word2code.get(w, word2code.get('<unk>', 0)) for w in tokens]
    cap_len = min(len(encoded), max_len)
    arr = np.zeros((max_len, 1), dtype='int64')
    arr[:cap_len, 0] = encoded[:max_len]
    return torch.LongTensor(arr).unsqueeze(0), torch.LongTensor([cap_len])

def main():
    # --- Paths ---
    data_path = "D:/GAN-codes/Deep_Fusion_GAN/data"
    encoder_weights_path = "D:/GAN-codes/Deep_Fusion_GAN/text_encoder_weights/text_encoder.pth"
    gen_weights_path = "../gen_weights"
    image_save_path = "../testing_images"
    os.makedirs(image_save_path, exist_ok=True)

    # --- Load vocabulary ---
    import pickle
    with open(os.path.join(data_path, "captions.pickle"), "rb") as f:
        _, _, code2word, word2code = pickle.load(f)

    # --- Load text encoder ---
    n_words = len(code2word)
    text_encoder = RNNEncoder.load(encoder_weights_path, n_words)
    text_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = text_encoder.to(device)

    # --- Load generator ---
    generator = Generator(n_channels=32, latent_dim=100).to(device)
    # Find latest generator weights
    gen_files = [f for f in os.listdir(gen_weights_path) if f.startswith("gen_") and f.endswith(".pth")]
    if not gen_files:
        raise FileNotFoundError("No generator weights found in gen_weights directory.")
    latest_gen = sorted(gen_files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
    generator.load_state_dict(torch.load(os.path.join(gen_weights_path, latest_gen), map_location=device))
    generator.eval()

    # --- User prompt ---
    prompt = input("Enter your text prompt: ").strip()
    if not prompt:
        print("No prompt entered.")
        return

    # --- Preprocess prompt ---
    captions, cap_lens = preprocess_caption(prompt, word2code)
    captions = captions.to(device)
    cap_lens = cap_lens.to(device)

    # --- Encode text ---
    with torch.no_grad():
        sent_emb = text_encoder(captions, cap_lens)

    # --- Generate image ---
    noise = torch.randn(1, 100).to(device)
    with torch.no_grad():
        fake_img = generator(noise, sent_emb)
    fake_img = fake_img.cpu()

    # --- Save image ---
    save_path = os.path.join(image_save_path, "sample_from_prompt.png")
    save_image(fake_img, save_path, normalize=True)
    print(f"Generated image saved to {save_path}")

    # --- Optionally display image ---
    img = fake_img[0]
    img = (img + 1) / 2  # [-1,1] to [0,1]
    img = transforms.ToPILImage()(img)
    img.show(title=f"Prompt: {prompt}")

if __name__ == "__main__":
    main()
