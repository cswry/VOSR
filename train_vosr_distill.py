import os
import yaml
import wandb, copy, glob
import logging
from collections import OrderedDict
from tqdm import tqdm
from models.lightningdit import LightningDiT
from pathlib import Path
from vosr import VOSR
from datasets import load_dataset
from dataloaders.realsr_dataset import TxtPairDataset
import gc


import torch, timm
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from safetensors.torch import save_file
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
my_torch_cache_root = 'preset/ckpts/torch_cache'
os.makedirs(my_torch_cache_root, exist_ok=True)
torch.hub.set_dir(my_torch_cache_root)
import torchvision
from torchvision import transforms
from torchvision.transforms import Normalize
from PIL import Image
import math, json
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pyiqa
import os, random
import numpy as np
from dataloaders.realesrgan_gpu import RealESRGAN_degradation

import os
import logging


torch.set_float32_matmul_precision("high")  # PyTorch 2.x
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_setDeviceAllocator'):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def load_model_weights_with_interpolation(accelerator, target_model, state_dict, model_name="model"):
    target_model_state = target_model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in target_model_state:
            if "pos_embed" in k and v.shape != target_model_state[k].shape:
                if accelerator.is_main_process:
                    print(f"[{model_name}] Resizing {k} from {v.shape} to {target_model_state[k].shape}")
                
                v_len = v.shape[1]
                target_len = target_model_state[k].shape[1]
                dim = v.shape[-1]
                
                src_size = int(math.sqrt(v_len))
                tgt_size = int(math.sqrt(target_len))
                
                v_img = v.reshape(1, src_size, src_size, dim).permute(0, 3, 1, 2)
                v_img = torch.nn.functional.interpolate(
                    v_img, size=(tgt_size, tgt_size), mode='bicubic', align_corners=False
                )
                v = v_img.permute(0, 2, 3, 1).reshape(1, tgt_size * tgt_size, dim)
            
            if "rope" in k or "freqs_cos" in k or "freqs_sin" in k:
                continue
                
            new_state_dict[k] = v
        else:
            if accelerator.is_main_process:
                print(f"[{model_name}] Skipping key {k} (not in current model)")

    msg = target_model.load_state_dict(new_state_dict, strict=False)
    if accelerator.is_main_process:
        print(f"[{model_name}] Loaded weights. Missing keys: {len(msg.missing_keys)}")
         

def _resolve_ckpt_dir(weight_file_path):
    """
    Given a weight file like .../checkpoint-00010000/clean_weights/ema_model.safetensors,
    walk up to find the checkpoint-XXXXXXXX directory and parse global_step.
    Returns: (ckpt_dir, global_step)  or  (None, 0).
    """
    cur = os.path.abspath(weight_file_path)
    for _ in range(5):
        cur = os.path.dirname(cur)
        basename = os.path.basename(cur)
        if basename.startswith("checkpoint-"):
            try:
                step = int(basename.split("-")[-1])
            except ValueError:
                step = 0
            return cur, step
    return None, 0


def find_latest_checkpoint(args):
    """
    Resolve checkpoint path and parse global step.
    - If args.resume_ckpt is a .safetensors / .pth file, trace up to find
      the checkpoint-XXXXXXXX directory for accelerator.load_state().
    - Otherwise auto-discover the latest checkpoint directory.
    Returns: (resume_dir, global_step)
    """
    checkpoint_dir = f"{args.output_dir}/checkpoints"
    resume_ckpt = args.resume_ckpt

    if resume_ckpt is not None and resume_ckpt != "":
        if os.path.isfile(resume_ckpt):
            ckpt_dir, step = _resolve_ckpt_dir(resume_ckpt)
            if ckpt_dir is not None:
                print(f"resume_ckpt file: {resume_ckpt} -> checkpoint dir: {ckpt_dir} (Step {step})")
                return ckpt_dir, step
            else:
                raise ValueError(
                    f"Cannot locate checkpoint-XXXXXXXX directory from: {resume_ckpt}"
                )
        elif os.path.isdir(resume_ckpt):
            try:
                step = int(os.path.basename(os.path.normpath(resume_ckpt)).split("-")[-1])
            except ValueError:
                step = 0
            return resume_ckpt, step
        else:
            print(f"resume_ckpt does not exist: {resume_ckpt}. Will try auto-discovery.")

    elif os.path.exists(checkpoint_dir):
        subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
        if len(subdirs) > 0:
            subdirs.sort(key=lambda x: int(x.split("-")[-1]))
            latest_ckpt_name = subdirs[-1]
            resume_path = os.path.join(checkpoint_dir, latest_ckpt_name)
            
            try:
                ckpt_files = os.listdir(resume_path)
                has_model_file = any(f in ckpt_files for f in ["pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json"])
                if not has_model_file:
                    raise ValueError(f"Checkpoint directory missing model files: {resume_path}")
                
                global_step = int(latest_ckpt_name.split("-")[-1])
                print(f"Found latest checkpoint: {resume_path} (Step {global_step})")
                return resume_path, global_step
            except (ValueError, Exception) as e:
                if len(subdirs) >= 2:
                    second_latest_ckpt_name = subdirs[-2]
                    resume_path = os.path.join(checkpoint_dir, second_latest_ckpt_name)
                    try:
                        global_step = int(second_latest_ckpt_name.split("-")[-1])
                        print(f"Latest checkpoint failed ({str(e)}). Using second-latest: {resume_path} (Step {global_step})")
                        return resume_path, global_step
                    except ValueError:
                        global_step = 0
                        print(f"Using second-latest checkpoint: {resume_path} (Step {global_step})")
                        return resume_path, global_step
                else:
                    try:
                        global_step = int(latest_ckpt_name.split("-")[-1])
                    except ValueError:
                        global_step = 0
                    print(f"Warning: latest checkpoint validation failed ({str(e)}), still using: {resume_path} (Step {global_step})")
                    return resume_path, global_step

    return None, 0



if is_wandb_available():
    import wandb
    wandb.login(key="654dc44a098866866f0c6fc7501c1a53182d7f2b")

logger = get_logger(__name__)

def filter_collate_fn(batch):
    """
    Collate that keeps only Tensor (and numeric) fields and drops strings.
    Prevents metadata like 'base_name' from breaking accelerate.
    """
    if not batch:
        return {}
        
    filtered_batch = []
    for item in batch:
        clean_item = {}
        for k, v in item.items():
            if isinstance(v, (torch.Tensor, int, float, np.ndarray)):
                clean_item[k] = v
        filtered_batch.append(clean_item)
    
    return torch.utils.data.default_collate(filtered_batch)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class SafeFileHandler(logging.FileHandler):
    """FileHandler that silently ignores flush errors on filesystems that don't support it."""
    def flush(self):
        try:
            super().flush()
        except OSError:
            pass

    def emit(self, record):
        try:
            super().emit(record)
        except OSError:
            pass

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), SafeFileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    Strips DDP (module.) and torch.compile (_orig_mod.) name prefixes.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        name = name.replace("_orig_mod.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def load_dinov2(args, device):
    if args.enc_type == 'dinov2b':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif args.enc_type == 'dinov2l':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    elif args.enc_type == 'dinov2g':
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        # encoder.load_state_dict(checkpoint, strict=False)

    del encoder.head

    encoder.head = torch.nn.Identity()

    def forward_with_features(self, x, masks=None):
        # print(f'++++> x_input size is {x.shape}')
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
        # print(f'++++> x_norm size is {x_norm.shape}')
        
        return features, x_norm[:, 1:]
    
    import types
    encoder.forward_with_features = types.MethodType(forward_with_features, encoder)

    encoder = encoder.to(device)
    encoder.eval()

    return encoder


def preprocess_raw_image(x, args):
    x = x / 255.
    x = torch.nn.functional.interpolate(x, args.dinov2_size, mode='bicubic').clip(0., 1.)
    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    return x


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset_config(path):
    txt_paths = []
    probs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 1:
                dataset_path, repeat = parts[0], 1
            elif len(parts) == 2:
                dataset_path, repeat = parts[0], int(parts[1])
            else:
                raise ValueError(f"Invalid line format: {line}")
            txt_paths.append(dataset_path)
            probs.append(repeat)
    return txt_paths, probs


def main(config_path):
    
    
    # Load configuration
    import os
    from argparse import Namespace
    config = load_config(config_path)
    args = Namespace(**config)



    if args.train_dataset_config is not None:
        txt_paths, probs = load_dataset_config(args.train_dataset_config)
        args.train_dataset_txt_paths_list = txt_paths
        args.train_dataset_prob_paths_list = probs

    weak_cond_strength_aelq = f'{args.weak_cond_strength_aelq_list[0]}-{args.weak_cond_strength_aelq_list[1]}'
    cfg_param = f's{args.cfg_scale}-r{args.cfg_ratio}-wc{weak_cond_strength_aelq}'

    if args.time_dist[0] == 'uniform':
        td = 'uni'
    elif args.time_dist[0] == 'lognorm':
        td = f'ln_{args.time_dist[1]}_{args.time_dist[2]}'

    resume = '_resume' if args.resume_ckpt is not None else ''

    downsample_scale = 8
    channel = 4 if args.ae_type == 'sd2' else 16
    args.exp_name = (
        f'ldit_distill_bs{args.train_batch_size*args.gradient_accumulation_steps:03d}'
        f'_{args.ae_type}f{downsample_scale}c{channel}'
        f'_size{args.resolution}'
        f'_ps{args.patch_size}_psr1'
        f'_d{args.dim}'
        f'_b{args.depth}'
        f'_h{args.num_heads}'
        f'_uw{args.u_weight}'
        f'_cfg{cfg_param}'
        f'_ts{args.t_start}e{args.t_end}'
        f'_edr{args.encdim_ratio}'
        f'_td{td}'
        f'_type{args.dataset_type}'
        f'_dist{args.distill_type}'
        f"{resume}"
        f"{args.suffix}"
    )

                    
    print(f'===> expname is {args.exp_name}')

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    
    logging_out_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(args.logging_dir))

    accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

    print(f'===> choose {len(args.layer_dinov2b_list)} layer fea for CA.')
    venc = load_dinov2(args, device=accelerator.device)
    print('===> loading vision encoder')
    checkpoint_dir = f"{args.output_dir}/checkpoints"  # Stores saved model checkpoints
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(args.output_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        # checkpoint_dir = f"{args.output_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(args.output_dir)
        logger.info(f"Experiment directory created at {args.output_dir}")

        iqa_lpips = pyiqa.create_metric('lpips', device='cuda')
        iqa_musiq = pyiqa.create_metric('musiq', device='cuda')
            
    if args.ae_type == "qwen":
        from models.qwenimage_vae2d import AutoencoderKLQwenImage2D
        model_ae = AutoencoderKLQwenImage2D.from_pretrained(args.ae_path)
    elif args.ae_type == "sd2":
        from diffusers import AutoencoderKL
        model_ae = AutoencoderKL.from_pretrained(args.ae_path, subfolder="vae")

    model_ae = model_ae.to(accelerator.device)
    model_ae.eval()
    
    # If passed along, set the training seed now.
    
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    base_channel = 4 if args.ae_type == 'sd2' else 16

    # Teacher: multi-step flow matching, same architecture as student, no auxiliary_time_cond
    model_tea = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2*base_channel,
        out_channels=base_channel,
        hidden_size=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        z_dims=args.enc_dim,
        encdim_ratio=args.encdim_ratio,
        auxiliary_time_cond=False,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rope=args.use_rope,
        use_rmsnorm=args.use_rmsnorm,
        wo_shift=args.wo_shift,
        num_fused_layers=len(args.layer_dinov2b_list), 
    )

    tea_ckpt_path = getattr(args, 'teacher_ckpt', None)
    if tea_ckpt_path is not None:
        from safetensors.torch import load_file as _load_file
        if tea_ckpt_path.endswith(".safetensors"):
            tea_state_dict = _load_file(tea_ckpt_path)
        else:
            tea_state_dict = torch.load(tea_ckpt_path, map_location="cpu")
        load_model_weights_with_interpolation(accelerator, model_tea, tea_state_dict, model_name="Teacher Model")
        if accelerator.is_main_process:
            print(f'=======> Loaded teacher weights from {tea_ckpt_path}')
    
    model_tea = model_tea.to(accelerator.device)
    model_tea.eval()
    requires_grad(model_tea, False)

    if accelerator.is_main_process:
        tea_total_params = sum(p.numel() for p in model_tea.parameters())
        logger.info(f"===========> Teacher parameters: {tea_total_params} ({tea_total_params / 1e9:.3f} B)")

    # Student: with auxiliary_time_cond
    model = LightningDiT(
        input_size=args.resolution // 8,
        patch_size=args.patch_size,
        in_channels=2*base_channel,
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

    total_params = sum(p.numel() for p in model.parameters())
    total_params_in_billion = total_params / 1e9
    if accelerator.is_main_process:
        logger.info(f"===========> Student parameters: {total_params} ({total_params_in_billion:.3f} B)")

    model.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model

    # Optimizer creation
    params_to_optimize = model.parameters()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        # power=args.lr_power,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    degradation = RealESRGAN_degradation('params_realsr.yml', device=accelerator.device)
    

    global_step = 0

    resume_path, current_step = find_latest_checkpoint(args)
    
    if args.seed is not None:
        seed_to_use = args.seed + current_step
        set_seed(seed_to_use)
        if accelerator.is_main_process:
            logger.info(f"Seed set to {seed_to_use} (Base {args.seed} + Step {current_step})")
    
    local_batch_size = int(args.train_batch_size // accelerator.num_processes)
    
    if accelerator.is_main_process:
        logger.info(f"===========> Batch Size Debug Info (DDP):")
        logger.info(f"  args.train_batch_size (global) = {args.train_batch_size}")
        logger.info(f"  accelerator.num_processes (total GPUs) = {accelerator.num_processes}")
        logger.info(f"  local_batch_size (per GPU) = {local_batch_size}")
        logger.info(f"  gradient_accumulation_steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size per GPU = {local_batch_size * args.gradient_accumulation_steps}")
        logger.info(f"  Total effective batch size = {local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
        logger.info(f"  Expected total batch size = {args.train_batch_size * args.gradient_accumulation_steps}")
        if local_batch_size * accelerator.num_processes * args.gradient_accumulation_steps != args.train_batch_size * args.gradient_accumulation_steps:
            logger.warning(f"  ⚠️ WARNING: Batch size mismatch! This may cause training issues.")
    
    num_workers = args.dataloader_num_workers
    prefetch_factor = 4 if args.dataloader_num_workers > 0 else None
    
    # Create dataset
    if args.dataset_type == 'txt':
        train_dataset = TxtPairDataset(split='train', args=args)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=prefetch_factor
        )
    elif args.dataset_type == 'webdataset':
        from dataloaders.realsr_dataset import build_webdataset_pipeline
        train_dataset = build_webdataset_pipeline(args, split='train')
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=local_batch_size,
            shuffle=False,  # IterableDataset
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if args.dataloader_num_workers > 0 else False,
            prefetch_factor=4,
            collate_fn=filter_collate_fn,
        )
    
    

    if accelerator.is_main_process:
        # logger.info(f"Dataset contains {len(train_dataset):,} images")
        to_tensor = transforms.ToTensor()             # [0,1]
        to_pil    = transforms.ToPILImage()


    vosr = VOSR(
        u_weight=args.u_weight,
        time_dist=args.time_dist,
        cfg_ratio=args.cfg_ratio,
        cfg_scale=args.cfg_scale,
        interp_type=args.interp_type,
        a=args.a,
        b=args.b,
        accelerator=accelerator,
        t_start=args.t_start,
        t_end=args.t_end,
        args=args
    )
    

    if args.use_ema:
        ema = copy.deepcopy(model)
        ema = ema.to(accelerator.device)
        requires_grad(ema, False)
        ema.eval()
        accelerator.register_for_checkpointing(ema)

    model, model_ae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, model_ae, optimizer, train_dataloader, lr_scheduler
    )

    
    
    if resume_path:
        global_step = current_step
        if accelerator.is_main_process:
            logger.info(f"Loading state from {resume_path}")
        accelerator.load_state(resume_path)
    elif args.pretrained_ckpt:
        pretrained_path = args.pretrained_ckpt
        if accelerator.is_main_process:
            print(f'=======> Loading pretrained weights from {pretrained_path}')
        
        from safetensors.torch import load_file

        if pretrained_path.endswith(".safetensors"):
            state_dict = load_file(pretrained_path)
        else:
            state_dict = torch.load(pretrained_path, map_location="cpu")

        unwrapped_model = accelerator.unwrap_model(model)
        load_model_weights_with_interpolation(accelerator, unwrapped_model, state_dict, model_name="Main Model")

        if args.use_ema:
            if accelerator.is_main_process:
                print("=======> Re-initializing EMA from loaded model weights")
            ema.load_state_dict(unwrapped_model.state_dict())

        global_step = 0
        if accelerator.is_main_process:
            print("=======> Optimizer and Scheduler reset. Global step set to 0.")
    else:
        global_step = 0
        
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        
        for key, value in tracker_config.items():
            if isinstance(value, list):
                tracker_config[key] = str(value)
        
        wandb_id_file = os.path.join(args.output_dir, "wandb_id.txt")
        wandb_id = None
        
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, 'r') as f:
                wandb_id = f.read().strip()
            print(f"=======> Resuming WandB run with ID: {wandb_id}")
        
        if wandb_id is None or wandb_id == "":
            import uuid
            wandb_id = uuid.uuid4().hex
            os.makedirs(args.output_dir, exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                f.write(wandb_id)
            print(f"=======> Created new WandB run with ID: {wandb_id}")

        wandb_init_kwargs = {
            "name": f"{args.exp_name}", 
            "dir": args.output_dir,
            "id": wandb_id,
            "resume": "allow"
        }
        
        accelerator.init_trackers(
            project_name=args.tracker_project_name, 
            config=tracker_config,
            init_kwargs={
                "wandb": wandb_init_kwargs 
            },
        )

    # Train!
    total_batch_size = args.train_batch_size 
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Instantaneous batch size per device = {local_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number of processes (GPUs) = {accelerator.num_processes}")
        logger.info(f"  Number of nodes = {accelerator.num_processes // 8 if accelerator.num_processes >= 8 else 1} (assuming 8 GPUs per node)")
        logger.info(f"  Distributed type = {accelerator.distributed_type}")
        logger.info(f"  ⚠️ DDP uses all-reduce for gradient synchronization")
        logger.info(f"  ⚠️ If communication bandwidth is high, communication time can be hidden by computation")
        logger.info(f" EXP = {args.exp_name}")
    # Training variables
    # global_step = 0
    losses = 0.0
    mse_losses = 0.0

    # initial_global_step = 0
    initial_global_step = global_step 

    # Training loop
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Training")
    epoch = 0
    unwrapped_model_ae = accelerator.unwrap_model(model_ae)
    
    
    unwrapped_venc = accelerator.unwrap_model(venc)
    if args.ae_type == "qwen":
        latents_mean = (
            torch.tensor(unwrapped_model_ae.config.latents_mean)
            .view(1, unwrapped_model_ae.config.z_dim, 1, 1)
            .to(unwrapped_model_ae.device, unwrapped_model_ae.dtype)
        )
        latents_std = 1.0 / torch.tensor(unwrapped_model_ae.config.latents_std).view(1, unwrapped_model_ae.config.z_dim, 1, 1).to(
            unwrapped_model_ae.device, unwrapped_model_ae.dtype
        )

    gc.disable()
    
    while global_step < args.max_train_steps:
        epoch += 1
        
        if epoch > 1:
            gc.collect()
            torch.cuda.empty_cache()
            
        for batch in train_dataloader:

            hq = batch["hq"].to(accelerator.device, non_blocking=True)
            with torch.no_grad():
                _, lq = degradation.degrade_process(hq, resize_bak=True)
                hq, lq = hq*2-1, lq*2-1

            with torch.no_grad():
                with accelerator.autocast():
                    raw_image = (0.5*lq + 0.5)*255 
                    raw_image_ = preprocess_raw_image(raw_image, args)
                    features, x_norm = unwrapped_venc.forward_with_features(raw_image_)
                    z = [v for k, v in features.items() if k.startswith('layer_')]
                    z[-1] = x_norm
                    z = [z[i] for i in args.layer_dinov2b_list]
                    
            with accelerator.accumulate(model):
                if global_step >= args.max_train_steps:
                    break
                    
                with torch.no_grad():
                    combined = torch.cat([lq, hq], dim=0)
                    if args.ae_type == 'qwen':
                        combined_latent = (unwrapped_model_ae.encode(combined).latent_dist.sample() - latents_mean) * latents_std
                    elif args.ae_type == 'sd2':
                        combined_latent = unwrapped_model_ae.encode(combined.to(unwrapped_model_ae.dtype)).latent_dist.sample() * unwrapped_model_ae.config.scaling_factor
                    lq, hq = combined_latent.chunk(2, dim=0)
                
                with accelerator.autocast():
                    if args.distill_type == 'shortcut':
                        loss, loss_backward = vosr.loss_fm_distill_shortcut_improved(model, lq, hq, z, model_tea=model_tea)
                    elif args.distill_type == 'rcgm':
                        loss, loss_backward = vosr.loss_fm_distill_rcgm_improved(model, lq, hq, z, model_tea=model_tea)
                    else:
                        raise ValueError(f"Invalid distill type: {args.distill_type}")

                # Backward and optimize
                accelerator.backward(loss_backward)
                
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, accelerator.unwrap_model(model), args.ema_decay)
                    # update_ema(ema, model)
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                lr_scheduler.step()              
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    accelerator.wait_for_everyone()
                    gc.collect()
                    torch.cuda.empty_cache()
                    ckpt_dir = f"{checkpoint_dir}/checkpoint-{global_step:08d}"
                    
                    accelerator.save_state(ckpt_dir)
                    
                    if accelerator.is_main_process:
                        import threading
                        
                        unwrapped_model = accelerator.unwrap_model(model)
                        model_state_cpu = {k: v.cpu().clone() for k, v in unwrapped_model.state_dict().items()}
                        ema_state_cpu = {k: v.cpu().clone() for k, v in ema.state_dict().items()} if args.use_ema else None
                        
                        def save_clean_weights(model_state, ema_state, save_dir):
                            os.makedirs(save_dir, exist_ok=True)
                            clean_model_dir = f"{save_dir}/clean_weights"
                            os.makedirs(clean_model_dir, exist_ok=True)
                            
                            save_file(model_state, f"{clean_model_dir}/model.safetensors")
                            if ema_state is not None:
                                save_file(ema_state, f"{clean_model_dir}/ema_model.safetensors")
                        
                        save_thread = threading.Thread(
                            target=save_clean_weights, 
                            args=(model_state_cpu, ema_state_cpu, ckpt_dir),
                            daemon=False  # avoid thread killed mid-write on process exit
                        )
                        save_thread.start()
                        
                    accelerator.wait_for_everyone()
                    torch.cuda.empty_cache()
                if global_step % args.infercence_steps == 1 and global_step > 0:
                    if accelerator.is_main_process:   
                        model.eval()
                        lq_dir   = Path(args.test_lq_dir)
                        gt_dir   = Path(args.test_gt_dir)

                        sr_dir   = Path(args.output_dir, "sr", f"{global_step:08d}")
                        sr_dir.mkdir(parents=True, exist_ok=True)


                        lpips_sum, musiq_sum, n_img = 0.0, 0.0, 0

                        with torch.no_grad():
                            for lq_path in sorted(lq_dir.glob("*.png")):
                                name = lq_path.stem
                                gt_path = gt_dir / f"{name}.png"
                                if not gt_path.exists():
                                    continue

                                # lq_img = Image.open(lq_path).convert("RGB")
                                with Image.open(lq_path) as img:
                                    lq_img = img.convert("RGB")
                                if hasattr(args, 'test_upscale') and args.test_upscale > 1:
                                    w, h = lq_img.size
                                    lq_img = lq_img.resize((w * args.test_upscale, h * args.test_upscale), Image.LANCZOS)

                                lq = to_tensor(lq_img).unsqueeze(0).to(accelerator.device)  # 0-1
                                lq = lq * 2.0 - 1.0                                                         # -1-1


                                with Image.open(gt_path) as img:
                                    rgb_img = img.convert("RGB")
                                    gt_01 = to_tensor(rgb_img).unsqueeze(0).to(accelerator.device)
                                # gt_m11 = gt_01 * 2.0 - 1.0                                                             # -1-1

                                with accelerator.autocast():
                                    raw_image = (0.5*lq + 0.5)*255 
                                    raw_image_ = preprocess_raw_image(raw_image, args)
                                    features, x_norm = unwrapped_venc.forward_with_features(raw_image_)
                                    z = [v for k, v in features.items() if k.startswith('layer_')]
                                    z[-1] = x_norm
                                    z = [z[i] for i in args.layer_dinov2b_list]


                                if args.ae_type == 'qwen':
                                    lq = (unwrapped_model_ae.encode(lq).latent_dist.sample() - latents_mean) * latents_std
                                    sr_m11 = vosr.sample_onestep(ema, lq, n_steps=args.infer_steps, venc_fea=z)
                                    sr_m11 = sr_m11 / latents_std + latents_mean
                                    sr_m11 = unwrapped_model_ae.decode(sr_m11, return_dict=False)[0].clamp(-1, 1)
                                elif args.ae_type == 'sd2':
                                    lq = unwrapped_model_ae.encode(lq).latent_dist.sample() * unwrapped_model_ae.config.scaling_factor
                                    sr_m11 = vosr.sample_onestep(ema, lq, n_steps=args.infer_steps, venc_fea=z)
                                    sr_m11 = sr_m11 / unwrapped_model_ae.config.scaling_factor
                                    sr_m11 = unwrapped_model_ae.decode(sr_m11, return_dict=False)[0].clamp(-1, 1)
                                    
                                sr_01 = (sr_m11 + 1.0) / 2.0
                                sr_01 = sr_01.clamp(0, 1)
                                to_pil(sr_01.squeeze(0).cpu()).save(sr_dir / f"{name}.png")
                                sr_01 = (sr_01 * 255).clamp(0, 255).byte()  # uint8 [0, 255]
                                sr_01 = sr_01.float() / 255.0      # float [0, 1]
                                
                                lpips_sum += iqa_lpips(gt_01, sr_01.clip(0, 1))
                                musiq_sum += iqa_musiq(sr_01.clip(0, 1))
                                n_img += 1

                        if n_img:
                            lpips_avg = lpips_sum.float().item() / n_img
                            musiq_avg = musiq_sum.float().item() / n_img

                        logger.info(
                            f"[Eval @ step {global_step}] "
                            f"LPIPS: {lpips_avg:.4f}  "
                            f"MUSIQ: {musiq_avg:.4f}"
                        )

                        accelerator.log(
                            {
                                "eval/lpips": lpips_avg,
                                "eval/musiq": musiq_avg,
                            },
                            step=global_step
                        )
                        model.train()
                    ###############################################################################
                    # End of added eval block                                                        #
                    ###############################################################################

            if global_step % 50 == 0:
                logs = {
                    "loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "v_loss": loss.item()
                }
                progress_bar.set_postfix(**logs)
                
            if accelerator.is_main_process and global_step % 250 == 0:
                logs = {
                    "loss": loss.detach().item(), 
                    "lr": lr_scheduler.get_last_lr()[0], 
                    "v_loss": loss.item()
                }
                accelerator.log(logs, step=global_step)

    gc.enable()
    gc.collect()
    
    # one_logger_callback_utils.on_train_end()            
    accelerator.wait_for_everyone()
    accelerator.end_training()
    # one_logger_callback_utils.on_app_end() 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_yml/vosr_fm_256_tar.yml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)