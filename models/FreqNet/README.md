## Environment setup
**Classification environment:** 
Ensure you have all necessary dependencies installed. If you haven’t installed them yet, run:
```sh
pip install -r requirements.txt
```
If you get a missing package error (e.g., tensorboardX), install it separately:
```sh
pip install tensorboardX
```

## Training the model
```sh
python train.py --dataroot /path/to/dataset --batch_size 32 --lr 0.001 --niter 50
```
- `--dataroot /path/to/dataset`: Path to the dataset that contains `train/`, `val/`, and `test/`.
- `--batch_size 32`: Adjust batch size based on your GPU memory.
- `--lr 0.001`: Set learning rate.
- `--niter 50`: Number of training epochs.

## Testing the model
```sh
python test.py --dataroot /path/to/dataset --model_path /path/to/checkpoints/last.pth
```
- `--model_path /path/to/saved_model.pth`: Path to the saved model after training (`checkpoints/last.pth`).
- `--dataroot /path/to/dataset`: Path to the dataset root (must contain `test/`).

## Monitor training
```sh
tensorboard --logdir=/path/to/checkpoints
```
Then, open your browser and go to:
http://localhost:6006

## Saving & Loading the Model
During training, models are automatically saved. The final model is stored as:
```sh
/checkpoints/last.pth
```
If you want to resume training:
```sh
python train.py --dataroot /path/to/dataset --model_path /path/to/checkpoints/last.pth
```