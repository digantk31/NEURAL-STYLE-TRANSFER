import os
import cv2
import numpy as np
torch
from torch import nn, optim
from torchvision import transforms, models
from torch.autograd import Variable

# -----------------------------------------------------------------------------
#               Neural Style Transfer Reimplementation
# -----------------------------------------------------------------------------
# This script applies artistic styles to photographs using PyTorch.
# It uses a pretrained VGG19 network to extract content and style features
# and optimizes a generated image to minimize a weighted combination of
# content, style, and total variation losses.
# -----------------------------------------------------------------------------

# ----------- Utility Functions -----------

def load_and_resize(image_path, target_size=None):
    """
    Read an image from disk and resize while preserving aspect ratio.
    Returns a float32 RGB array normalized to [0,1].
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if target_size:
        h, w = img_rgb.shape[:2]
        if isinstance(target_size, int):
            new_h = target_size
n            new_w = int(w * (target_size / h))
        else:
            new_w, new_h = target_size
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return img_rgb


def tensor_preprocess(img_array, device, mean_rgb, std_rgb):
    """
    Convert a numpy image array to a normalized PyTorch tensor on the target device.
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),              # convert HWC [0,1] to CxHxW [0,1]
        transforms.Normalize(mean=mean_rgb, std=std_rgb)
    ])
    return preprocess(img_array).unsqueeze(0).to(device)


def tensor_to_image(tensor, mean_rgb, std_rgb):
    """
    Convert a normalized tensor back to a uint8 HxWxC image.
    """
    t = tensor.clone().detach().cpu().squeeze(0)
    for c in range(3):
        t[c] = t[c] * std_rgb[c] + mean_rgb[c]
    np_img = np.clip(t.numpy().transpose(1,2,0), 0, 1) * 255
    return np_img.astype(np.uint8)


def save_image(np_img, save_path):
    """
    Save an RGB numpy image (HxWxC) to disk in JPEG format.
    """
    bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr)


def gram_matrix(feature_map):
    """
    Compute the Gram matrix from a feature activation map.
    """
    b, c, h, w = feature_map.size()
    fm = feature_map.view(b, c, h * w)
    return torch.bmm(fm, fm.transpose(1, 2)) / (c * h * w)


def total_variation_loss(image_tensor):
    """
    Encourage spatial smoothness in the generated image.
    """
    diff_x = torch.abs(image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :-1]).sum()
    diff_y = torch.abs(image_tensor[:, :, 1:, :] - image_tensor[:, :, :-1, :]).sum()
    return diff_x + diff_y


# ----------- Model Setup -----------
class StyleContentModel(nn.Module):
    """
    Wrap a pretrained VGG19 to extract intermediate activations for content and style.
    """
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.selected_layers = content_layers + style_layers
        self.slices = nn.ModuleList()
        prev = 0
        for layer_idx in sorted(self.selected_layers):
            slice = nn.Sequential(*list(vgg.children())[prev:layer_idx+1])
            self.slices.append(slice)
            prev = layer_idx+1
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features


def run_neural_style_transfer(
        content_path,
        style_path,
        output_dir,
        img_size=400,
        content_weight=1e5,
        style_weight=1e4,
        tv_weight=1.0,
        num_steps=500):
    """
    Main entry: perform style transfer and save final result.
    """
    # Mean and std for ImageNet as used in torchvision
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare images
    content_np = load_and_resize(content_path, target_size=img_size)
    style_np   = load_and_resize(style_path, target_size=img_size)

    content_img = tensor_preprocess(content_np, device, mean, std)
    style_img   = tensor_preprocess(style_np, device, mean, std)

    # Initialize generated as a copy of content
    generated = Variable(content_img.clone(), requires_grad=True)

    # Define which layers to use
    content_layers = [21]  # conv4_2
    style_layers   = [0, 5, 10, 19, 28]  # conv1_1, conv2_1, ...

    model = StyleContentModel(content_layers, style_layers).to(device)

    # Get feature maps
    content_features = model(content_img)
    style_features   = model(style_img)
    style_grams = [gram_matrix(sf) for sf in style_features]

    # Optimizer
    optimizer = optim.LBFGS([generated], max_iter=num_steps)

    iteration = [0]
    def closure():
        optimizer.zero_grad()
        gen_features = model(generated)

        # Content loss
        c_loss = nn.MSELoss()(gen_features[len(style_layers)], content_features[len(style_layers)])

        # Style loss
        s_loss = 0
        for i in range(len(style_layers)):
            gm_generated = gram_matrix(gen_features[i])
            s_loss += nn.MSELoss()(gm_generated, style_grams[i])
        s_loss /= len(style_layers)

        # Total variation
        tv_loss = total_variation_loss(generated)

        total = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss
        total.backward()

        iteration[0] += 1
        if iteration[0] % 50 == 0 or iteration[0] == num_steps:
            print(f"Step {iteration[0]}/{num_steps}: total={total.item():.2f}, c={c_loss.item():.2f}, s={s_loss.item():.2f}, tv={tv_loss.item():.2f}")
        return total

    optimizer.step(closure)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_image = tensor_to_image(generated, mean, std)
    save_path = os.path.join(output_dir, 'styled_result.jpg')
    save_image(output_image, save_path)

    print(f"Style transfer complete. Image saved at: {save_path}")
    return save_path


if __name__ == '__main__':
    # Example usage
    base_dir = os.getcwd()
    content_file = os.path.join(base_dir, 'data', 'content.jpg')
    style_file   = os.path.join(base_dir, 'data', 'style.jpg')
    result_dir   = os.path.join(base_dir, 'output')

    run_neural_style_transfer(
        content_path=content_file,
        style_path=style_file,
        output_dir=result_dir,
        img_size=400,
        content_weight=1e5,
        style_weight=3e4,
        tv_weight=1.0,
        num_steps=500
    )