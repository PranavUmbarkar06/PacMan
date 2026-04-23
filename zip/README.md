# PacManDRL

`PacManDRL` is a small Deep Q-Learning project where Pac-Man learns to clear the maze while a rule-based ghost tries to catch it.

## Files You Will Use

- `train.py` trains the agent and saves the model.
- `visualise.py` opens the pygame viewer and shows a trained model playing.
- `p_brain.pth` is the default trained model file that `visualise.py` loads.

## Recommended Python Version

This project's existing virtual environment uses Python `3.11.9`, so Python `3.11` is the safest choice.
## For project evaluation 
FOR AI PROJECT WORK 
DON'T TRAIN AGAIN TAKES TOO LONG

```powershell
python -m venv venv;.\venv\Scripts\activate; pip install -r requirements.txt; python visualise.py
```

simply run this in the terminal 

## Setup

1. Create a virtual environment:
```powershell
python -m venv venv
```

2. Activate it:

```powershell
.\venv\Scripts\activate
```

3. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

4. Install the main dependencies:

```powershell
pip install numpy pygame pillow
```

5. Install PyTorch:

If you have an NVIDIA GPU with CUDA 12.1:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you want the CPU-only version:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

6. Optional: install the full environment from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

`requirements.txt` also includes notebook/debug packages, so it is larger than the minimum needed just to run the project.

## Train The Agent

Train for 400 episodes:

```powershell
python train.py --episodes 400
```

This saves the trained model to `p_brain.pth` and also writes a checkpoint file `p_brain.ckpt`.

## View The Trained Model

Open the visualiser with the default saved model:

```powershell
python visualise.py
```

Useful options:

```powershell
python visualise.py --model-path p_brain.pth
python visualise.py --fps 8
python visualise.py --episodes 5
python visualise.py --cpu
```

What these options do:

- `--model-path` chooses which `.pth` file to load.
- `--fps` controls how fast the game logic runs.
- `--episodes` limits how many episodes are shown. `0` means it keeps running until you close the window.
- `--cpu` forces inference on CPU instead of CUDA.

## Typical Workflow

```powershell
python train.py --episodes 400
python visualise.py --model-path p_brain.pth
```

## Notes

- Keep `pacman.png` and `ghost.png` in the project root so the viewer can load the sprites.
- If `visualise.py` says the model file is missing, train first or pass the correct file with `--model-path`.
- `main.py` is also a viewer, but `visualise.py` is the newer and better script to use for watching the agent.
