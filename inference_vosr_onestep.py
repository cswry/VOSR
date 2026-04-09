import os
import sys
import gc
import types
import glob
import argparse
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize
from safetensors.torch import load_file
from tqdm import tqdm

torch.hub.set_dir('preset/ckpts/torch_cache')
sys.path.append(os.getcwd())

from models.lightningdit import LightningDiT
from models.light_decoder import LightDecoder
from vosr import VOSR

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

SD2_LWDECODER_PATH = 'preset/ckpts/sd21_lwdecoder.pth'
SD2_AE_PATH = 'preset/ckpts/stable-diffusion-2-1-base'
QWEN_AE_PATH = 'preset/ckpts/Qwen-Image-vae-2d'


IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}

def list_lq_images(input_path):
    p = os.path.normpath(input_path)
    if os.path.isfile(p):
        return [p]
    if os.path.isdir(p):
        imgs = [
            os.path.join(p, f) for f in sorted(os.listdir(p))
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        return imgs
    return []


def all_expected_outputs_exist(image_paths, out_dirs):
    if not image_paths:
        return False
    for img_path in image_paths:
        name = os.path.basename(img_path)
        for d in out_dirs:
            if not os.path.isfile(os.path.join(d, name)):
                return False
    return True


def adain_color_fix(target, source):
    from torchvision.transforms.functional import to_tensor, to_pil_image
    target_tensor = to_tensor(target).unsqueeze(0)
    source_tensor = to_tensor(source).unsqueeze(0)
    eps = 1e-5
    target_mean = torch.mean(target_tensor, dim=[2, 3], keepdim=True)
    target_std = torch.std(target_tensor, dim=[2, 3], keepdim=True) + eps
    source_mean = torch.mean(source_tensor, dim=[2, 3], keepdim=True)
    source_std = torch.std(source_tensor, dim=[2, 3], keepdim=True) + eps
    target_tensor = (target_tensor - target_mean) / target_std * source_std + source_mean
    return to_pil_image(torch.clamp(target_tensor[0], 0, 1))


def wavelet_color_fix(target, source):
    import cv2
    target_np = np.array(target).astype(np.float32) / 255.0
    source_np = np.array(source.resize(target.size, Image.LANCZOS)).astype(np.float32) / 255.0
    sigma = 5
    source_low = cv2.GaussianBlur(source_np, (0, 0), sigma)
    target_low = cv2.GaussianBlur(target_np, (0, 0), sigma)
    target_high = target_np - target_low
    result = np.clip(source_low + target_high, 0, 1) * 255.0
    return Image.fromarray(result.astype(np.uint8))


def load_config_with_cli(ckpt_path, parser):
    cur = os.path.abspath(ckpt_path)
    json_path = None
    for _ in range(5):
        candidate = os.path.join(cur, "args.json")
        if os.path.isfile(candidate):
            json_path = candidate
            break
        cur = os.path.dirname(cur)
    if json_path is None:
        print(f"Warning: args.json not found near {ckpt_path}, using default CLI args only.")
        return parser.parse_args()
    with open(json_path, 'r') as f:
        json_args = json.load(f)
    cli_args = parser.parse_args()
    combined_args = argparse.Namespace()
    for k, v in json_args.items():
        setattr(combined_args, k, v)
    for k, v in vars(cli_args).items():
        if v is not None:
            setattr(combined_args, k, v)
        elif not hasattr(combined_args, k):
            setattr(combined_args, k, v)
    return combined_args


def _clear_stale_distributed_env():
    for k in (
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE',
        'GROUP_RANK', 'ROLE_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_MAX_RESTARTS',
    ):
        os.environ.pop(k, None)


def _inference_accelerator_stub():
    _clear_stale_distributed_env()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return types.SimpleNamespace(device=dev)


def preprocess_raw_image(x, args):
    x = x / 255.
    x = F.interpolate(x, args.dinov2_size, mode='bicubic').clip(0., 1.)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


def load_dinov2(args, device):
    if args.enc_type == 'dinov2b':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif args.enc_type == 'dinov2l':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif args.enc_type == 'dinov2g':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
    del encoder.head
    encoder.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        features = {}
        layer_indices = list(range(len(self.blocks)))
        if isinstance(x, list):
            return self.forward_features_list(x, masks)
        x = self.prepare_tokens_with_masks(x, masks)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                features[f'layer_{i}'] = x[:, 1:]
        x_norm = self.norm(x)
        return features, x_norm[:, 1:]

    encoder.forward_with_features = types.MethodType(forward_with_features, encoder)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_venc_features(venc, lq_tensor, args):
    with torch.no_grad():
        raw_image = (0.5 * lq_tensor + 0.5) * 255
        raw_image_ = preprocess_raw_image(raw_image, args)
        features, x_norm = venc.forward_with_features(raw_image_)
        z = [v for k, v in features.items() if k.startswith('layer_')]
        z[-1] = x_norm
        z = [z[i] for i in args.layer_dinov2b_list]
    return z


def _encode_latent(vae, x, args, device):
    if args.ae_type == 'qwen':
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1).to(device)
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1).to(device)
        return (vae.encode(x).latent_dist.sample() - latents_mean) * latents_std, latents_mean, latents_std
    elif args.ae_type == 'sd2':
        return vae.encode(x).latent_dist.sample() * vae.config.scaling_factor, None, None


def _decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder=None):
    if args.ae_type == 'sd2':
        sr_u = sr_latent / vae.config.scaling_factor
        return light_decoder(sr_u).clamp(-1, 1)
    elif args.ae_type == 'qwen':
        sr_latent = sr_latent / latents_std + latents_mean
        return vae.decode(sr_latent, return_dict=False)[0].clamp(-1, 1)


def tiled_inference(
    model, vosr_model, vae, venc, lq_tensor, args,
    tile_size=512, tile_overlap=64, device='cuda',
    light_decoder=None, sample_fn=None,
):
    b, c, h, w = lq_tensor.shape
    output = torch.zeros((b, c, h, w), device=device, dtype=lq_tensor.dtype)
    count = torch.zeros((b, 1, h, w), device=device, dtype=lq_tensor.dtype)

    z_fea_full = get_venc_features(venc, lq_tensor, args) if venc is not None else None

    stride = tile_size - tile_overlap
    h_steps = list(range(0, max(h - tile_size + 1, 1), stride))
    if h > tile_size and (h_steps[-1] + tile_size) < h:
        h_steps.append(h - tile_size)
    if h <= tile_size:
        h_steps = [0]
    w_steps = list(range(0, max(w - tile_size + 1, 1), stride))
    if w > tile_size and (w_steps[-1] + tile_size) < w:
        w_steps.append(w - tile_size)
    if w <= tile_size:
        w_steps = [0]

    for h_idx in h_steps:
        for w_idx in w_steps:
            h_end = min(h_idx + tile_size, h)
            w_end = min(w_idx + tile_size, w)
            lq_tile = lq_tensor[:, :, h_idx:h_end, w_idx:w_end]

            with torch.no_grad():
                lq_latent, latents_mean, latents_std = _encode_latent(vae, lq_tile, args, device)

                z_fea_tile = None
                if z_fea_full is not None:
                    z_fea_tile = []
                    for z_layer in z_fea_full:
                        B, N, C = z_layer.shape
                        feat_size = int(math.sqrt(N))
                        fh_s = int(h_idx / h * feat_size)
                        fh_e = max(int(h_end / h * feat_size), fh_s + 1)
                        fw_s = int(w_idx / w * feat_size)
                        fw_e = max(int(w_end / w * feat_size), fw_s + 1)
                        z_2d = z_layer.view(B, feat_size, feat_size, C)
                        z_fea_tile.append(z_2d[:, fh_s:fh_e, fw_s:fw_e, :].reshape(B, -1, C))

                sr_latent = sample_fn(model, lq_latent, n_steps=args.infer_steps, venc_fea=z_fea_tile)
                sr_tile = _decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder)

            output[:, :, h_idx:h_end, w_idx:w_end] += sr_tile
            count[:, :, h_idx:h_end, w_idx:w_end] += 1.0

    return output / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--input_image', '-i', type=str, required=True)
    parser.add_argument('--output_dir', '-o', type=str, required=True)
    parser.add_argument("--upscale", '-u', type=int, default=1)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--tile_size", type=int, default=0, help='Tile size for diffusion inference (0 to disable)')
    parser.add_argument("--tile_overlap", type=int, default=32)
    parser.add_argument('--infer_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force_rerun', action='store_true')
    parser.add_argument('--config', type=str, default='')

    temp_args, _ = parser.parse_known_args()
    args = load_config_with_cli(temp_args.checkpoint, parser)

    if args.infer_steps is None:
        args.infer_steps = 1

    seed = args.seed
    image_paths = list_lq_images(args.input_image)
    if not image_paths:
        print("No LQ images found. Exit.")
        return

    run_stem = f'{args.ae_type}_steps{args.infer_steps}_seed{seed}_{args.distill_type}'
    out_dir = f'{args.output_dir}/{run_stem}'

    if (
        not getattr(args, 'force_rerun', False)
        and not os.environ.get('INFER_NO_SKIP', '').lower() in ('1', 'true', 'yes')
        and all_expected_outputs_exist(image_paths, [out_dir])
    ):
        print(f"Skip: all {len(image_paths)} outputs already exist. Use --force_rerun to override.")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = _inference_accelerator_stub()
    print(f"Loading model from {args.checkpoint}...")
    print(f"Inference steps: {args.infer_steps}, Seed: {seed}")

    # 1. Load VAE
    ae_path = getattr(args, 'ae_path', QWEN_AE_PATH if args.ae_type == 'qwen' else SD2_AE_PATH)
    if args.ae_type == "qwen":
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        vae = AutoencoderKLQwenImage2D.from_pretrained(ae_path)
    elif args.ae_type == "sd2":
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(ae_path, subfolder="vae")
    vae.to(device).eval()

    # 2. Load LightDecoder (sd2 only)
    light_decoder = None
    if args.ae_type == 'sd2':
        print(f"Loading LightDecoder from {SD2_LWDECODER_PATH}...")
        ckpt = torch.load(SD2_LWDECODER_PATH, map_location="cpu")
        dec_config = ckpt["config"]
        light_decoder = LightDecoder(
            in_channels=dec_config["in_channels"],
            out_channels=dec_config["out_channels"],
            block_out_channels=tuple(dec_config["block_out_channels"]),
            layers_per_block=dec_config["layers_per_block"],
        )
        light_decoder.load_state_dict(ckpt["model_state_dict"])
        light_decoder.to(device).eval()

    # 3. Load Vision Encoder (dinov2)
    venc = load_dinov2(args, device)

    # 4. Load DiT Model (distilled student with auxiliary_time_cond)
    base_channel = 4 if args.ae_type == 'sd2' else 16
    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2 * base_channel,
        out_channels=base_channel,
        hidden_size=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio,
        auxiliary_time_cond=True,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list),
    )

    search_dirs = [
        os.path.join(args.checkpoint, "clean_weights"),
        os.path.join(args.checkpoint, "checkpoints"),
        args.checkpoint,
    ]
    weight_path = None
    for d in search_dirs:
        for name in ("ema_model.safetensors", "model.safetensors"):
            candidate = os.path.join(d, name)
            if os.path.isfile(candidate):
                weight_path = candidate
                break
        if weight_path:
            break
    if weight_path is None:
        hits = glob.glob(os.path.join(args.checkpoint, "**/*.safetensors"), recursive=True)
        if not hits:
            raise FileNotFoundError(f"No .safetensors found under {args.checkpoint}")
        weight_path = hits[0]

    print(f"Loading weights from {weight_path}")
    state_dict = load_file(weight_path) if weight_path.endswith(".safetensors") else torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    model.forward = model.forward_flexible

    # 5. Initialize VOSR
    vosr_model = VOSR(
        time_dist=args.time_dist,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        interp_type=args.interp_type,
        accelerator=accelerator,
        t_start=args.t_start,
        t_end=args.t_end,
        args=args,
    )

    sample_fn = vosr_model.sample_onestep

    # 6. Inference Loop
    os.makedirs(out_dir, exist_ok=True)
    args.output_dir = out_dir
    to_tensor = transforms.ToTensor()
    print(f"Processing {len(image_paths)} images...")

    for idx, img_path in enumerate(tqdm(image_paths, file=sys.stderr)):
        img_name = os.path.basename(img_path)
        raw_img = Image.open(img_path).convert("RGB")
        w, h = raw_img.size
        target_w, target_h = w * args.upscale, h * args.upscale
        if idx == 0:
            print(f"Output size: {target_w}x{target_h}, input size: {w}x{h}")

        input_img = raw_img.resize((target_w, target_h), Image.BICUBIC)
        lq = to_tensor(input_img).unsqueeze(0).to(device) * 2.0 - 1.0

        try:
            if args.tile_size > 0:
                sr_tensor = tiled_inference(
                    model, vosr_model, vae, venc, lq, args,
                    tile_size=args.tile_size, tile_overlap=args.tile_overlap,
                    device=device, light_decoder=light_decoder, sample_fn=sample_fn,
                )
            else:
                with torch.no_grad():
                    lq_latent, latents_mean, latents_std = _encode_latent(vae, lq, args, device)
                    z_fea = get_venc_features(venc, lq, args)
                    sr_latent = sample_fn(model, lq_latent, n_steps=args.infer_steps, venc_fea=z_fea)
                    sr_tensor = _decode_latent(vae, sr_latent, args, latents_mean, latents_std, light_decoder)

            sr_img = transforms.ToPILImage()(sr_tensor[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                sr_img = adain_color_fix(sr_img, input_img)
            elif args.align_method == 'wavelet':
                sr_img = wavelet_color_fix(sr_img, input_img)
            sr_img.save(os.path.join(out_dir, img_name))

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
