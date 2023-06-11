import librosa
import torch
from train import TextDecoder
from tqdm import tqdm
from data_handling.clotho_dataset import get_clotho_loader
from eval_metrics import evaluate_metrics
import argparse
import torch
from clap_model.ase_model import ASE
from ruamel import yaml
import librosa
import os 
import yaml
from dotmap import DotMap


def get_config():

    with open('settings/settings.yaml', 'r') as f:

        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)
    
    with open("settings/inference.yaml", "r") as f:
        clap_config = yaml.safe_load(f)

    return config, clap_config





def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='trained_models/clotho_baseline/best_model.pt')
    parser.add_argument('--clap_path', default='/home/theokouz/data/WavCaps/cnn14-bert.pt')
    parser.add_argument('--dataset', default='clotho')
    parser.add_argument('--eval_dir', default='/home/theokouz/data/clotho/wavforms/evaluation')
    parser.add_argument('--method', default='ad')
    parser.add_argument('--mem', default='data/clotho.json')

    args = parser.parse_args()

    return args


















def Decoding(model,clip_features):


    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    entry_length = 30
    temperature = 0.1
    tokens = None
    for i in range(entry_length):
        outputs = model.decoder(inputs_embeds=embedding_cat)
        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)
        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==102:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:


        output_list = list(tokens.squeeze().cpu().numpy())
        output = clap.text_encoder.tokenizer.decode(torch.tensor(output_list[1:-1]))
    except:
        output = 'None'
    return output






import json
def construct_support_memory(text_json, clap):
    with open(text_json, 'r') as f:
        data = json.load(f)
    text_features = []
    captions = []
    batch_size = 1000
    clap.eval()
    for i in tqdm(range(0,len(data[:])//batch_size)):
        
        texts = data[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            text_feature = clap.encode_text(texts)
            text_features.append(text_feature)
            captions.extend(texts)

    text_features = torch.cat(text_features,dim=0)
    text_features /= text_features.norm(dim=-1,keepdim=True).float()    
    return text_features





def eval(test_data, model, clap, args, use_beam, mapper = None):
    evaluation_dir = args.eval_dir
    captions_gt = []
    captions_pred = []
    text_features = construct_support_memory(args.mem, clap)
    text_features = text_features.to(device)

    for _, eval_batch in tqdm(enumerate(test_data), total=len(test_data)):
        _, target_dicts, file_names = eval_batch
        for file_name, target in zip(file_names, target_dicts):
            audio_path = os.path.join(evaluation_dir, file_name)
            target['file_name'] = file_name

            captions_gt.append(target)

            with torch.no_grad():
                audio_data, _ = librosa.load(audio_path, sr=32000) 
                audio_data = torch.tensor(audio_data).unsqueeze(0).to(device)
                audio_embed = clap.encode_audio(audio_data)

                audio_embed /= audio_embed.norm(dim=-1, keepdim=True)
                # Decode with the audio embedding
                if args.method == 'ad':
                    prefix_embedding = torch.tensor(audio_embed).to(device)
                # Decode with the nearest neighbor
                elif args.method == 'nnd':
                    audio_embed = torch.tensor(audio_embed).to(device)
                    sim = audio_embed@text_features.T.float()
                    sim = (sim*100).softmax(dim=-1)
                    nearest_neighbor = torch.argmax(sim).item()
                    prefix_embedding = text_features[nearest_neighbor].unsqueeze(0)
                    prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
                # Project audio embedding to Memory and decode with the projection
                else:
                    audio_embed = torch.tensor(audio_embed).to(device)
                    sim = audio_embed@text_features.T.float()
                    sim = (sim*100).softmax(dim=-1)
                    prefix_embedding = sim@text_features.float()
                    prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
                prediction = Decoding(model, prefix_embedding)
            captions_pred.append(
                {
                    'file_name': file_name,
                    'caption_predicted': prediction
                }
            )
    metrics = evaluate_metrics(prediction_file=captions_pred, reference_file= captions_gt)
    for metric, values in metrics.items():
        print(f'greddy: {metric:<7s}: {values["score"]:7.4f}')

    return metrics




if __name__ == '__main__':

    args = argument_parser()
    config, clap_config = get_config()
    clap = ASE(clap_config)
    cp_path = args.clap_path
    cp = torch.load(cp_path)
    clap.load_state_dict(cp['model'])
    clap.eval()
    device = 'cuda:3'
    clap.to(device)
    model = TextDecoder(prefix_size=1024)
    test_data = get_clotho_loader(config, dataset_name=args.dataset)
    weights_path = args.model_path
    model.load_state_dict(torch.load(weights_path,map_location= torch.device('cpu')))
    model = model.to(device)
    model = model.eval()
    eval(test_data, model, clap, args, use_beam=False)





