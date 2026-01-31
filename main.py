import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader, random_split
import wandb

wandb.init(project="mnist-ae")

# set up data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tensor_transform = transforms.ToTensor()

batch_size = 256
MNIST_data = datasets.MNIST(root = "./data",
							train = True,
							download = True,
							transform = tensor_transform)

MNIST_test = datasets.MNIST(root="./data",
                            train=False,
                            download=True,
                            transform=tensor_transform
)

train_size = int(0.8 * len(MNIST_data))
val_size = len(MNIST_data) - train_size

MNIST_train, MNIST_val = random_split(MNIST_data, [train_size, val_size])

MNIST_train_loader = torch.utils.data.DataLoader(dataset = MNIST_train,
							                    batch_size = batch_size,
								                shuffle = True)

MNIST_val_loader = torch.utils.data.DataLoader(dataset = MNIST_val,
                                                batch_size = batch_size,
                                                shuffle = False)

MNIST_test_loader = torch.utils.data.DataLoader(dataset = MNIST_test,
                                                batch_size = batch_size,
                                                shuffle = False)

from math import e
mse = torch.nn.MSELoss()

# train functions
def loss_func(model, x, reg_func=None, coeff=1e-3):
    output = model(x)
    err = mse(output['imgs'], x)
    logpx_z = -1.0 * torch.sum(err)

    if reg_func is not None:
      reg = reg_func(output)
    else:
      reg = 0.0

    return -1.0 * torch.mean(logpx_z + coeff * reg)

def train(train_dataloader, val_dataloader, model, loss_func, optimizer, epochs):
    losses = []
    val_losses =[]
    # training loop
    for epoch in tqdm(range(epochs), desc='Epochs'):
        model.train()
        running_loss = 0.0
        batch_progress = tqdm(train_dataloader, desc='Train Batches', leave=False)

        for iter, (images, labels) in enumerate(batch_progress):
            batch_size = images.shape[0]
            # images = images.reshape(batch_size, -1).to(device)
            images = images.to(device)
            loss = loss_func(model, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
 
            wandb.log({"train_loss": loss.item(), 
                        "epoch": epoch, 
                        "step": iter + epoch * len(train_dataloader)})

        avg_loss = running_loss / len(train_dataloader)
        losses.append(avg_loss)
        # tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}\n')
        
        # validation
        model.eval()
        val_loss = 0.0
        batch_progress = tqdm(val_dataloader, desc='Val Batches', leave=False)
        with torch.no_grad():  
            for iter, (images, labels) in enumerate(batch_progress):
                batch_size = images.shape[0]
                # images = images.reshape(batch_size, -1).to(device)
                images = images.to(device)
                loss = loss_func(model, images)
                val_loss += loss.item() 
                
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
        tqdm.write(f'----\nEpoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}\n')
        
        if epoch % 5 == 0:
            original = images[0]
            reconstructed = model(images[0].unsqueeze(0))['imgs']
            orig_numpy = original.cpu().detach().numpy().squeeze()
            recon_numpy = reconstructed.cpu().detach().numpy().squeeze()
            wandb.log({
                "original": wandb.Image(orig_numpy), 
                "reconstructed": wandb.Image(recon_numpy)})

    return losses, val_losses

# eval functions
# def plot_latent_images(model, n, digit_size=28):
#     grid_x = np.linspace(-2, 2, n)
#     grid_y = np.linspace(-2, 2, n)

#     image_width = digit_size * n
#     image_height = digit_size * n
#     image = np.zeros((image_height, image_width))

#     for i, yi in enumerate(grid_x):
#         for j, xi in enumerate(grid_y):
#             z = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
#             with torch.no_grad():
#                 x_decoded = model.decode(z)
#             digit = x_decoded.view(digit_size, digit_size).cpu().numpy()
#             image[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit

#     plt.figure(figsize=(10, 10))
#     plt.imshow(image, cmap='Greys_r')
#     plt.axis('Off')
#     plt.show()


# def eval(model):
#     original_imgs = torch.cat([MNIST_test[i][0] for i in range(5)])
#     with torch.no_grad():
#       res = model(original_imgs.reshape(5, -1).to(device))
#       reconstructed_imgs = res['imgs']
#       reconstructed_imgs = reconstructed_imgs.cpu().reshape(*original_imgs.shape)

#     fig, axes = plt.subplots(5, 2, figsize=(10, 25))

#     for i in range(5):
#         original_image = original_imgs[i].reshape(28, 28)
#         axes[i, 0].imshow(original_image, cmap='gray')
#         axes[i, 0].set_title(f'Original Image {i+1}')
#         axes[i, 0].axis('off')

#         reconstructed_image = reconstructed_imgs[i].reshape(28, 28)
#         axes[i, 1].imshow(reconstructed_image, cmap='gray')
#         axes[i, 1].set_title(f'Reconstructed Image {i+1}')
#         axes[i, 1].axis('off')

#     plt.tight_layout()
#     plt.show()

# train
from ae import AE

def loss_AE(model, x):
    reconstructed = model(x)['imgs']
    return mse(reconstructed, x)

image_shape = MNIST_train[0][0].shape
print(image_shape)
input_dim = torch.prod(torch.tensor(image_shape)).item()
print("input_dim: ", input_dim)

# we r decreasing hidden_dims here to force compression/a bottleneck
# some popular models like vggnet/resnet do increasing hidden_dims bc as they decrease the img size, they want to preserve features to extract
hidden_dims = [128, 32, 16, 2]

ae = AE(input_dim, hidden_dims).to(device)
print(ae)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

print(count_parameters(ae))

#---------------------
optimizer_ae = torch.optim.Adam(ae.parameters(),
                                lr = 1e-3,
                                weight_decay = 1e-8)

epochs = 20
#---------------------

log_ae = train(MNIST_train_loader,MNIST_val_loader, ae, loss_AE, optimizer_ae, epochs)


torch.save(ae.state_dict(), "ae_model.pth")
