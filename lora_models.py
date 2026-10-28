"""Strict project-specific model loaders and deterministic image preprocessing."""
import hashlib
import importlib.util
import os
import re
import sys
from pathlib import Path

os.environ['XFORMERS_DISABLED'] = '1'
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from lora_layers import use_sdpa


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hash(root, paths):
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).replace('\\', '/').encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Encoder(nn.Module):
    def __init__(self, branch, options):
        super().__init__()
        self.branch = branch
        self.options = dict(options)
        self.size = int(options.get('size', 512 if branch == 'chexfound' else 224))
        self.identity = {'branch': branch, 'runner_models_sha256':file_hash(__file__),
                         'runner_layers_sha256':file_hash(Path(__file__).with_name('lora_layers.py')),
                         'torch_version':str(torch.__version__)}
        if branch == 'raddino':
            from transformers import AutoModel, AutoImageProcessor
            model_id = options.get('rad_model', 'microsoft/rad-dino')
            revision = options.get('rad_revision', '110cbc18d5133582e320b43d53bf5c44e410c936')
            if not re.fullmatch(r'[0-9a-f]{40}',revision):
                raise ValueError('RAD revision must be an immutable 40-character commit SHA')
            self.net = AutoModel.from_pretrained(model_id, revision=revision)
            self.net.set_attn_implementation('sdpa') if hasattr(self.net, 'set_attn_implementation') else None
            proc = AutoImageProcessor.from_pretrained(model_id, revision=revision)
            self.dim = self.net.config.hidden_size
            self.identity.update(model=model_id, revision=revision, config=self.net.config.to_dict())
            self.transform = T.Compose([T.Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC),
                                        T.ToTensor(), RepeatChannels(), T.Normalize(proc.image_mean, proc.image_std)])
        elif branch == 'chexfound':
            from omegaconf import OmegaConf
            repo = Path(options['chexfound_repo']).resolve()
            sys.path.insert(0, str(repo))
            from chexfound.models import build_model_from_cfg
            cfg = OmegaConf.load(options['chexfound_config'])
            cfg.student.block_chunks = 0
            self.net, self.dim = build_model_from_cfg(cfg, only_teacher=True)
            # Map the checkpoint: avoid eagerly materializing the unused DINO/iBOT heads in RAM.
            raw = torch.load(options['chexfound_weights'], map_location='cpu', weights_only=True, mmap=True)['teacher']
            state = {}
            for key, value in raw.items():
                key = key.removeprefix('module.')
                if key.startswith('backbone.'):
                    key = key[len('backbone.'):]
                    key = re.sub(r'^blocks\.\d+\.(\d+)\.', r'blocks.\1.', key) if re.match(r'^blocks\.\d+\.\d+\.', key) else key
                    if key in state:
                        raise ValueError(f'Duplicate remapped checkpoint key: {key}')
                    state[key] = value
            self.net.load_state_dict(state, strict=True)
            use_sdpa(self.net)
            # Import only transforms; the upstream data package also imports training dependencies.
            transforms = import_file('_lora_cf_transforms', repo / 'chexfound/data/transforms.py')
            self.transform = transforms.make_classification_eval_transform(resize_size=self.size, crop_size=self.size)
            self.identity.update(weights=file_hash(options['chexfound_weights']), config=file_hash(options['chexfound_config']),
                                 source=source_hash(repo, list((repo/'chexfound').rglob('*.py'))))
        elif branch == 'chexzero':
            repo = Path(options['chexzero_repo']).resolve()
            CLIP = import_file('_lora_chexzero_model', repo / 'model.py').CLIP
            raw = torch.load(options['chexzero_weights'], map_location='cpu', weights_only=True)
            state = raw.get('state_dict', raw)
            state = {k.removeprefix('module.'): v for k, v in state.items()}
            width, channels, patch, patch_w = state['visual.conv1.weight'].shape
            n = state['visual.positional_embedding'].shape[0]
            grid = round((n-1)**.5)
            if channels != 3 or patch != patch_w or grid*grid+1 != n:
                raise ValueError('Unsupported CheXzero visual architecture')
            self.size = grid * patch
            self.dim = state['text_projection'].shape[1]
            layers = len([k for k in state if k.startswith('visual.transformer.resblocks.') and k.endswith('.attn.in_proj_weight')])
            tw = state['ln_final.weight'].shape[0]
            tl = len({k.split('.')[2] for k in state if k.startswith('transformer.resblocks.')})
            self.net = CLIP(embed_dim=self.dim, image_resolution=self.size, vision_layers=layers, vision_width=width,
                            vision_patch_size=patch, context_length=state['positional_embedding'].shape[0],
                            vocab_size=state['token_embedding.weight'].shape[0], transformer_width=tw,
                            transformer_heads=tw//64, transformer_layers=tl)
            for key in ('input_resolution', 'context_length', 'vocab_size'):
                state.pop(key, None)
            self.net.load_state_dict(state, strict=True)
            self.transform = T.Compose([T.Resize((self.size,self.size),interpolation=InterpolationMode.BICUBIC),
                                        T.ToTensor(), RepeatChannels(),
                                        T.Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
            self.identity.update(weights=file_hash(options['chexzero_weights']), source=file_hash(repo/'model.py'),
                                 patch=patch, layers=layers)
        else:
            raise ValueError(branch)
        self.net.float().eval().requires_grad_(False)
        self.identity.update(size=self.size, dim=self.dim, preprocessing='fourbranch-compatible-v1')

    def forward(self, x):
        if self.branch == 'raddino':
            return self.net(pixel_values=x).last_hidden_state[:, 0].float()
        if self.branch == 'chexfound':
            return self.net.get_intermediate_layers(x, n=1, return_class_token=True)[0][1].float()
        return self.net.encode_image(x).float()


class RepeatChannels:
    def __call__(self, x):
        return x.repeat(3, 1, 1)
