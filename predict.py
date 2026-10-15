"""Run binary vegetation segmentation on one RGB tile or a directory of tiles."""
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image
import torch

from model import VegetationNet

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples')
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--weights', type=Path, help='Override the weights path in config.json.')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:0')
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding='utf-8'))
    threshold = float(config['threshold'])
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1.')
    weights = args.weights or args.config.resolve().parent / config['weights']
    if not weights.is_file():
        raise FileNotFoundError(f'Weights not found: {weights}. Pass --weights /path/to/vegetation_model.pt.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = VegetationNet(base_channels=config['base_channels'])
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    model.to(device).eval()

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    if args.input.is_dir():
        images = sorted(p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    elif args.input.is_file() and args.input.suffix.lower() in extensions:
        images = [args.input]
    else:
        raise ValueError(f'Input must be an image file or a directory: {args.input}')
    if not images:
        raise ValueError(f'No supported images found in {args.input}')
    if len({p.stem for p in images}) != len(images):
        raise ValueError('Use unique image stems so output directories do not overlap.')

    for path in images:
        with Image.open(path) as image:
            rgb = np.array(image.convert('RGB'))
        if list(rgb.shape[:2]) != config['input_size']:
            raise ValueError(f'{path.name}: expected {config["input_size"]} pixels, got {list(rgb.shape[:2])}.')
        x = torch.from_numpy(rgb.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(device)
        probability = model(x).sigmoid()[0, 0].cpu().numpy()
        mask = probability >= threshold
        out = args.output / path.stem
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'probability.npy', probability)
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / 'mask.png')
        overlay = rgb.copy()
        overlay[mask] = (.65 * rgb[mask] + .35 * np.array([0, 255, 128])).astype(np.uint8)
        Image.fromarray(overlay).save(out / 'overlay.png')
        print(f'{path.name} -> {out.resolve()} ({device})')


if __name__ == '__main__':
    main()
