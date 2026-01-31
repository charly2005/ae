This repository contains a PyTorch implementation of an Autoencoder trained on the MNIST dataset. The model compresses 28x28 digit images into a 2-dimensional latent space and reconstructs them.

Clone the repository.
```bash
git clone "https://github.com/charly2005/ae.git"
```

To run, first have conda or miniconda installed.
Then create and activate a virtual environment.
```bash
conda create --name venv python=3.9
conda activate venv
```

Next, install the requirements.
```bash
pip install -r requirements.txt
```

Now run the training script. This took < 2min with a L40S GPU, CPU training may take up to hours.
```bash
python main.py
```

Make sure you also have a wandb account (https://wandb.ai/) and have created an api key.
If you don't want use wandb you can comment out every line referencing it.

Example terminal output.
```bash
torch.Size([1, 28, 28])
input_dim:  784
AE(
  (relu): ReLU()
  (sigmoid): Sigmoid()
  (encoder_list): ModuleList(
    (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (2): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): Conv2d(16, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  )
  (decoder_list): ModuleList(
    (0): ConvTranspose2d(2, 16, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (1): ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (2): ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (3): ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
  )
)
86179

wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from /home/ubuntu/.netrc.
wandb: Currently logged in as: cyao030 (cyao030-boston-university) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.24.1
wandb: Run data is saved locally in /home/ubuntu/ae/wandb/run-20260131_175810-9qnal8k9
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run dandy-forest-17
wandb: ⭐️ View project at https://wandb.ai/cyao030-boston-university/mnist-ae
wandb: 🚀 View run at https://wandb.ai/cyao030-boston-university/mnist-ae/runs/9qnal8k9
torch.Size([1, 28, 28])
input_dim:  784
AE(
  (relu): ReLU()
  (sigmoid): Sigmoid()
  (encoder_list): ModuleList(
    (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (2): Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): Conv2d(16, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  )
  (decoder_list): ModuleList(
    (0): ConvTranspose2d(2, 16, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (1): ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (2): ConvTranspose2d(32, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
    (3): ConvTranspose2d(128, 1, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
  )
)
86179
----                                                                                               
Epoch [1/20], Train Loss: 0.0968, Val Loss: 0.0626                                                 

----                                                                                               
Epoch [2/20], Train Loss: 0.0562, Val Loss: 0.0518                                                 

----                                                                                               
Epoch [3/20], Train Loss: 0.0497, Val Loss: 0.0484                                                 
                                             
----

...(truncated)                                            

----                                                                                               
Epoch [18/20], Train Loss: 0.0410, Val Loss: 0.0410                                                

----                                                                                               
Epoch [19/20], Train Loss: 0.0408, Val Loss: 0.0409                                                

----                                                                                               
Epoch [20/20], Train Loss: 0.0406, Val Loss: 0.0407                                                

Epochs: 100%|██████████████████████████████████████████████████████| 20/20 [01:06<00:00,  3.32s/it]
wandb: 
wandb: 🚀 View run dandy-forest-17 at: https://wandb.ai/cyao030-boston-university/mnist-ae/runs/9qnal8k9
wandb: Find logs at: wandb/run-20260131_175810-9qnal8k9/logs
```

Results:
Original image:
<img src="assets/original.png" width="500" alt"">

First epoch attempt:
<img src="assets/reconstructed1.png" width="500" alt="">

Last epoch attempt: 
<img src="assets/reconstructed20.png" width="500" alt="">

Training & Validation Graph:
<img src="assets/assets.png" width="500" alt="">