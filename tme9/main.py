import sys
sys.path.append("./src")
import torch
from tp9 import FolderText, get_imdb_data, Model_Q1, Model_Q2, Model_Q3
from lit import MyLitDataModule, LitModel
import pytorch_lightning as pl



# from tools import *



word2id, embeddings, trainset, testset = get_imdb_data(embedding_size=50)
vocab_size, embedding_dim = embeddings.shape

data = MyLitDataModule(FolderText.collate, trainset, testset, testset, batch_size=32)
data.setup()

trainer = pl.Trainer(max_epochs=100, progress_bar_refresh_rate=30)



# print("########################################################## MODEL 1 ##########################################################")
# litmodel_q1 = LitModel(
#     backbone=Model_Q1(vocab_size, embedding_dim, 2, static_embeding_weight=embeddings), 
#     learning_rate=2e-4
# )
# trainer.fit(litmodel_q1, data)
# trainer.test(litmodel_q1)

print("########################################################## MODEL 2 ##########################################################")
litmodel_q3 = LitModel(
    backbone=Model_Q3(vocab_size, embedding_dim, 2, static_embeding_weight=embeddings), 
    learning_rate=2e-4
)
trainer.fit(litmodel_q3, data)
trainer.test(litmodel_q3)


# print("########################################################## MODEL 3 ##########################################################")
# litmodel_q3 = LitModel(
#     backbone=Model_Q3(vocab_size, embedding_dim, 2, static_embeding_weight=embeddings),
#     learning_rate=2e-4
# )
# trainer.fit(litmodel_q3, data)
# trainer.test(litmodel_q3)




