import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3
from scipy import linalg

@torch.no_grad()
def compute_inception_score(images, cuda=True, batch_size=8, splits=1):
    # images: (N, 3, H, W), values in [-1, 1]
    N = images.size(0)
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
    
    try:
        preds = []
        # Check for NaN values in input images
        has_nan = torch.isnan(images).any() or torch.isinf(images).any()
        if has_nan:
            print("Warning: Input images contain NaN or Inf values for IS calculation")
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
            
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(device)
            batch = (batch + 1) / 2  # [-1,1] to [0,1]
            batch = torch.clamp(batch, min=0, max=1)  # Ensure valid image range
            batch = up(batch)
            
            with torch.no_grad():
                logits = model(batch)
                
            # Handle logits carefully to prevent NaN in softmax
            logits = torch.clamp(logits, min=-50, max=50)  # Prevent extreme values
            probs = F.softmax(logits, dim=1).cpu().numpy()
            
            # Check for NaN in probabilities
            if np.isnan(probs).any() or np.isinf(probs).any():
                print("Warning: NaN or Inf values detected in IS probabilities")
                probs = np.nan_to_num(probs, nan=1.0/1000, posinf=1.0/1000, neginf=1.0/1000)
                
            preds.append(probs)
            
        preds = np.concatenate(preds, axis=0)
        
        # Calculate IS with numerical safety
        split_scores = []
        for k in range(splits):
            part = preds[k * (N // splits): (k+1) * (N // splits), :]
            py = np.mean(part, axis=0) + 1e-10  # Add small epsilon to avoid zeros
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i] + 1e-10  # Add small epsilon to avoid zeros
                scores.append(np.sum(pyx * (np.log(pyx) - np.log(py))))
            split_scores.append(np.exp(np.mean(scores)))
        
        return float(np.mean(split_scores))
    except Exception as e:
        print(f"Error in IS calculation: {e}")
        return 1.0  # Return baseline value on error

@torch.no_grad()
def compute_fid(real_images, fake_images, cuda=True, batch_size=8):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    
    try:
        model = inception_v3(pretrained=True, transform_input=False).to(device)
        # Properly modify for feature extraction - we need the 2048-dim vector before classification
        model.eval()
        model.fc = torch.nn.Identity()
        model.aux_logits = False  # Turn off auxiliary outputs
        
        up = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        
        def get_activations(images):
            # Check for NaN values in input images
            if torch.isnan(images).any() or torch.isinf(images).any():
                print("Warning: Input images contain NaN or Inf values")
                images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=-1.0)
                
            activations = []
            for i in range(0, images.size(0), batch_size):
                batch = images[i:i+batch_size].to(device)
                batch = (batch + 1) / 2  # [-1,1] → [0,1]
                batch = torch.clamp(batch, 0, 1)  # Ensure values are in valid range
                batch = up(batch)
                
                features = model(batch)
                activations.append(features.cpu())
            
            return np.concatenate([act.numpy() for act in activations], axis=0)
        
        # Process images and check for issues
        act1 = get_activations(real_images)
        act2 = get_activations(fake_images)
        
        # Check for NaN values in activations
        if np.isnan(act1).any() or np.isnan(act2).any():
            print("Warning: Activations contain NaN values")
            return float('inf')
            
        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
        
        # Safe computation of FID
        try:
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
            return float(fid)
        except Exception as e:
            print(f"Error in FID calculation: {e}")
            return float('inf')
    except Exception as e:
        print(f"Exception in FID calculation: {e}")
        return float('inf')
