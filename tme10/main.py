import sys
sys.path.append('./src')

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import LitModel
from tp10 import *



word2id, embeddings, trainset, testset, = get_imdb_data(embedding_size=50)


trainloader = torch.utils.data.DataLoader(
    trainset, num_workers=4, batch_size=32, shuffle=True, collate_fn=FolderText.collate)
testloader = torch.utils.data.DataLoader(
    testset, num_workers=4, batch_size=32, shuffle=False, collate_fn=FolderText.collate)


model1 = MeanEmbedding(
    vocab_size=len(word2id),
    embedding_dim=50,
    n_class=2,
    static_embeding_weight=embeddings
)


litModel1 = LitModel(backbone=model1, num_steps_per_epochs=len(
    trainloader), num_epochs=250, learning_rate=1e-1)

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=f"./lightning_logs",
    filename='vit-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}',
    mode='max',
)

lr_monitor = LearningRateMonitor(logging_interval='step')


trainer = pl.Trainer(progress_bar_refresh_rate=30, max_epochs=250, callbacks=[
                    checkpoint_callback, lr_monitor])

trainer.fit(litModel1, train_dataloader=trainloader,
            val_dataloaders=testloader)
