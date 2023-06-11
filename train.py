import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import argparse
import json
from typing import Tuple
import random
import json
import random
from clap_model.ase_model import ASE
from ruamel import yaml

# from test_clap import evaluation

class ClapTextDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)
    
    def pad_tokens(self, tokens: int):
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        clap_tokens = self.captions_tokens[item]
        return clap_tokens, self.captions[item]

    def __init__(self, data_path: str, model, gpt2_type: str = "gpt2"):
        self.max_seq_len = 22
        self.model = model
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)

        with open(data_path, 'r') as f:
            self.captions = json.load(f)
        random.shuffle(self.captions)
        self.captions_tokens = []
        for caption in self.captions[:]:
            self.captions_tokens.append(
                self.pad_tokens(
                    torch.tensor(self.model.text_encoder.tokenizer(caption)['input_ids'], dtype=torch.int64)
                    )
                )

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class TextDecoder(nn.Module):

    def __init__(self,prefix_size: int = 1024):
        super(TextDecoder, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is trained from scratch
        with open('pickles/decoder_config.pkl','rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clap_project = MLP((prefix_size,self.embedding_size))

    def forward(self, clap_features,gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clap = self.clap_project(clap_features)
        embedding_clap = embedding_clap.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clap,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out



import math
def noise_injection(x, device, variance=0.016, modality_offset=None, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)
    x = x + (torch.randn(x.shape, device=device) * std)  
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=1)



def train_decoder(
        dataset: ClapTextDataset,
        args,
        device,
        clap,
        modality_gap,
        output_dir: str = ".",
        output_prefix: str = ""
    ):

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    SEED=42
    torch.cuda.manual_seed_all(SEED)
    
    model = TextDecoder()


    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0)
    model.to(device)
    
    optimizer = AdamW(model.parameters(),lr=args.lr, weight_decay=args.wd)

    train_dataloader = DataLoader(dataset,batch_size=batch_size,drop_last=True)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        model.train()
        loss_token_save,ac_save= 0,0
        print(f">>> Training epoch {epoch}")
        progress = tqdm(total=int(len(train_dataloader)/10), desc=output_prefix)

        # dist.barrier()
        for idx,(clap_tokens, captions) in tqdm(enumerate(train_dataloader)):
            clap_tokens = clap_tokens.to(device)

            with torch.no_grad():

                captions = list(captions)
                feature_text = torch.tensor(clap.encode_text(captions))
                feature_text = feature_text.to(device)

                feature_text = noise_injection(
                    feature_text, 
                    variance=args.noise, 
                    device=device, 
                    modality_offset=modality_gap
                )



            outputs = model(feature_text.float(),clap_tokens)
            logits = outputs
            
            logits = logits.logits

            logits = logits[:,: -1]
            clap_tokens = clap_tokens.flatten()
            logits = logits.reshape(-1, logits.shape[-1])
            
            loss_token = loss_ce(logits, clap_tokens)
            ac=((logits.argmax(1)==clap_tokens)*(clap_tokens>0)).sum()/(clap_tokens>0).sum()
            optimizer.zero_grad()
            loss_all = loss_token
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            
            if(idx+1) %10 ==0:
                progress.set_postfix({"loss_token": loss_token_save/10.0,"acc_token":ac_save/10.0})
                progress.update()
                loss_token_save,ac_save= 0,0
            else:
                loss_token_save += loss_token.item()
                ac_save += ac.item()

        progress.close()
        torch.save(
            model.state_dict(),
            os.path.join(output_dir, f"epoch-{epoch}.pt"),
        )



    return model







def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/audiocaps.json')
    parser.add_argument('--clap_path', default='/home/theokouz/data/WavCaps/cnn14-bert.pt')
    parser.add_argument('--out_dir', default='./clotho_model')
    parser.add_argument('--prefix', default='./clotho_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='clotho', help='clotho or audiocaps')
    parser.add_argument('--modality_gap_path', default=None, help='pickled modality gap vector')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--device', default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--warmup', type=float, default=100)
    parser.add_argument('--wd', type=float, default=0.02)
    parser.add_argument('--noise', type=float, default=0.001)


    args = parser.parse_args()
    device = 'cuda:{}'.format(args.device)

    with open("settings/inference.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.modality_gap_path is not None:
        with open(args.modality_gap_path, 'rb') as handle:
            modality_gap = pickle.load(handle)
            modality_gap = modality_gap.to(device)
    else:
        modality_gap = None



    clap = ASE(config)
    clap.to(device)
    cp_path = args.clap_path

    cp = torch.load(cp_path)
    clap.load_state_dict(cp['model'])
    clap.eval()

    dataset = ClapTextDataset(args.data, model=clap)



    train_decoder(dataset, args, device, 
        clap=clap,
        modality_gap=modality_gap,
        output_dir=args.out_dir,
        output_prefix=args.prefix
    )



if __name__ == '__main__':
    main()

