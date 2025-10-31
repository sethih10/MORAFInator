import torch
from torch.utils.data import DataLoader


from src.components import Encoder, Decoder
from src.tokenization import get_tokenizer
from src.dataloader.data_loader import get_datasets, afm_collate_fn



def inference(encoder, decoder, img, device = torch.device('cpu')): 

    encoder.eval()
    decoder.eval()


    img = img.to(device)


    features, hiddens = encoder(img)

    results = decoder.decode(features, hiddens)
    return results
    