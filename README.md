## Install

Use Python 3.10–3.12. From this directory, run:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -r requirements.txt
```

## Load weights

Download [`vegetation_model.pt` from Google Drive](https://drive.google.com/file/d/1dKVrZddA9P7YpfBDdvcNgfV34DiSZSoK/view?usp=drivesdk) and place it in `weights/`.

## Run the two examples

```bash
python predict.py
```

The two images in `examples/` are processed automatically. Use `--device cpu` to run on CPU; otherwise an available CUDA device is selected automatically.

Each example produces a subdirectory in `outputs/` containing `mask.png` (0 = background, 255 = vegetation), `probability.npy` (float32 probabilities), and `overlay.png` (prediction overlaid on the input).
