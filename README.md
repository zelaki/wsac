## Weakly Supervised Automaed Audio Captioning via Text Only Training

<img src="https://github.com/zelaki/wsac/blob/main/data/main_fig-1.png" width=90% height=90%>


### About 
In recent years, datasets of paired audio and captions have enabled 
remarkable success in automatically generating descriptions for audio clips, namely Automated Audio Captioning (AAC). 
However, it is labor-intensive and time-consuming to collect a sufficient number of paired audio and captions.
Motivated by the recent advanced in Contrastive Language-Audio Pretraining (CLAP),
we propose a weakly-supervised approach to train an AAC model assuming only
text data and a pre-trained CLAP model, alleviating the need for
paired target data. Our approach leverages the similarity between
audio and text embeddings in CLAP. During training, we learn to
reconstruct the text from the CLAP text embedding, and during inference, we decode using the audio embeddings.
To mitigate the modality gap between the audio and text embeddings we employ
strategies to bridge the gap during training and inference stages.
We evaluate our proposed method on the Clotho and AudioCaps
dataset demonstrating its ability to achieve up to 80% of the 
performance attained by fully supervised approaches trained on paired target data.


### Requirements
Clone, create environment and install dependencies:

```
git clone https://github.com/zelaki/wsac.git && cd wsac
conda create --name wsac --file requirements.txt
conda activate wsac
```

### Prepare Training Data
Prepare json files with captions from Clotho and AudioCaps datasets
```
cd data
python3 csv2json.py /path/to/clotho/train.csv /path/to/audiocaps/train.csv
```

### Download CLAP model.

Download CLAP model trained on [WavCaps](https://github.com/XinhaoMei/WavCaps/tree/master) dataset from [link](https://drive.google.com/drive/folders/1MeTBren6LaLWiZI8_phZvHvzz4r9QeCD).





### Training

Train the model using train.py script:
```
python3 train.py --data data/clotho.json --out_dir trained_models/clotho  
```
The configurations / hyperparameters  are the following:
```
Arguments:
  --data Path to training data captions (clotho.json / audiocaps.json)
  --clap_path Path to clap model weights
  --out_dir Dir to save trained models
  --prefix  Prefix for saved filenames
  --modality_gap_path Path to pickled modality gap vector
  --epochs Number of epochs to train
  --bs Bathc size
  --lr Learning rate
  --warmup Number of warm-up steps
  --wd Wight decay factor
  --noise Noise Variance 
```


- Noise Injection training as described in [[1]](https://arxiv.org/pdf/2211.00575.pdf), set --noise equal to the desired noise variance. 
- Embedding Shift training set --modality_gap_path to the path of a pickled modality gab vector.



### Evaluation
Evaluated the trained models using evaluation.py script
```
evaluation.py --model_path path/to/trained/model.pt --clap_path path/to/clap/model.pt --dataset clotho --eval_dir path/to/testset/waveforms 
```
The arguments of the evaluation script are the following:

```
Arguments:
  --model_path          Path to trained model
  --clap_path           Path to clap model
  --dataset             Dataset name (clotho/audiocaps)
  --eval_dir            Path to dirctory with waveforms from the test set
  --method              What inference stradegy top use for decoding (ad/nnd/pd)
  --mem                 Path to json with text to use for Memory construciton. 
 ```
 
