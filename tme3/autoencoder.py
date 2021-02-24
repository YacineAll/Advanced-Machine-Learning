import sys
import argparse
import os
from datetime import datetime
sys.path.append('./src')
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from models import AutoEncoder, HighwayNet
from utils import *
from torch.utils.tensorboard import SummaryWriter
import torchvision


def main():
    
    save = "./experiments"
    log_dir = "./experiments"
    input_path = "./data//mnist"
    batch_size = 16
    lr = 1e-3
    latent_size = 12
    n_iter = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     

    checkpoint_dir = f"{save}/checkpoints/autoencoder"     
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device.type == 'cuda':
        log_dir = f"{log_dir}/logs/autoencoder/cuda"
    else:
        log_dir = f"{log_dir}/logs/autoencoder/cpu"
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = f'{checkpoint_dir}/checkpoint_' + datetime.now().strftime('%d_%m_%Y_%H:%M:%S')
    
    writer = SummaryWriter(log_dir)


    writer = SummaryWriter(log_dir)
    

    data = MNIST(transform=True, test_size=0.1, train_batch_size = batch_size, input_path=input_path)
    traindata, valdata, testdata = data.data()
    train, val, test = data.loader()

    n = 300
    x, labels = testdata[np.random.randint(0, len(testdata), n)]
    images, labels = torch.from_numpy(x.reshape(n, 1, 28,28)), torch.from_numpy(labels).to(device)
    img_grid = torchvision.utils.make_grid(images)
    # matplotlib_imshow(img_grid, one_channel=True)
    writer.add_image(f'{n}_mnist_images', img_grid)
    
    images, labels = images.to(device), labels.to(device)
    
    model = AutoEncoder(28*28, latent_size)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    model = model.to(device)
    
    writer.add_graph(model, images.view(len(images),28*28))
    
    losses = train_autoencoder(
        train, 
        test, 
        model, 
        criterion, 
        optimizer, 
        device, 
        checkpoint_path, 
        writer, 
        n_iter=n_iter
    )
    
    

    with torch.no_grad():
        projection = model.encodeur(images.view(len(images),28*28))
    
    writer.add_embedding(
        projection,
        metadata=labels,
        label_img=images
    )

if __name__ == "__main__":
    main()
