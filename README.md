## Language2Pose:Natural Language Grounded Pose Forecasting
### [[Paper](https://arxiv.org/pdf/1907.01108.pdf)][[Webpage](http://chahuja.com/language2pose)]


There are 5 steps to running this code
* Python Virtual Environment and dependencies
* Data download and preprocessing
* Training
* Sampling
* Rendering

----

PS: The implementation of one of the baselines, proposed by Lin et al.[[1]](#references), was not publicly available and hence we make use of **our [implementation](#sampling)** to generate all the results and animations marked as Lin et al. Due to the differences in training hyperparameters, dataset and experiments, the numbers reported for Lin et al. in our paper differ from the ones in the original paper [[1]](#references).

PS: This repo, at the moment, is functional at best. Feel free to create issues/pull requests however you see fit. 

----
### Python Virtual Environment
Anaconda is recommended to create the virtual environment
```sh
conda create -f env.yaml
source activate torch
```

To handle the logistics of saving/loading models [pycasper](https://github.com/chahuja/pycasper) is used
```sh
git clone https://github.com/chahuja/pycasper
cd src 
ln -s ../pycasper/pycasper .
cd ..
```

----
### Data 
#### Download
We use KIT Motion-Language Dataset which can be downloaded [here](https://motion-annotation.humanoids.kit.edu/dataset)

```sh
wget https://motion-annotation.humanoids.kit.edu/downloads/4/2017-06-22.zip
mkdir dataset/kit-mocap
unzip 2017-06-22.zip -d dataset/kit-mocap
rm 2017-06-22.zip 
```

#### Download Word2Vec binaries
Download the binary file [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) and place it in `src/s2v`

#### Pre-trained Models
Download pretrained models [here](https://drive.google.com/drive/folders/1hPOAhvZpmcAdZJgKrH8aEQPUe1MLfU0j?usp=sharing) and place it in `src/save`

#### Preprocessing
```sh
python data/data.py -dataset KITMocap -path2data ../dataset/kit-mocap
```

#### Rendering Ground Truths
```sh
python render.py -dataset KITMocap -path2data ../dataset/kit-mocap/new_fke -feats_kind fke
```

#### Calculating mean+variance for Z-Normalization
```sh
python dataProcessing/meanVariance.py -mask '[0]' -feats_kind rifke -dataset KITMocap -path2data ../dataset/kit-mocap -f_new 8
```

----
### Training
We train the models using a script `train_wordConditioned.py` (Pardon the misnomer; initially it was supposed to be word conditioned pose forecasting but then I ended up adding sentence conditioned pose forecasting as well and was too lazy to change the filename.)

All the arguments (and their corresponding help texts) used for training can be found in src/argsUtils.py (PS: Some of them might be deprecated, but I have not removed them in case it breaks any of the other code that I might have written in the experimentation phase. Please raise an issue/ or send me an email if you have any clarification questions about any of the arguments). It would be good to stick to the args used in the examples if you want to play with the models in the paper.

- JL2P
```sh
python train_wordConditioned.py -batch_size 100 -cpk jl2p -curriculum 1 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke -losses "['SmoothL1Loss']" -lr 0.001 -mask "[0]" -model Seq2SeqConditioned9 -modelKwargs "{'hidden_size':1024, 'use_tp':False, 's2v':'lstm'}" -num_epochs 1000 -path2data ../dataset/kit-mocap -render_list subsets/render_list -s2v 1 -save_dir save/model/ -tb 1 -time 16 -transforms "['zNorm']" 
```

`-modelKwargs` need some explaination as they could vary based on the model

```sh
hidden_size: size of the joint embedding
use_tp: use a trajectory predictor [1]. False for JL2P models
s2v: sentence to vector model ('lstm' or 'bert')
```

- Lin et. al. [1]
```sh
python train_seq2seq.py -batch_size 100 -cpk lin -curriculum 0 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke -losses "['MSELoss']" -lr 0.001 -mask "[0]" -model Seq2Seq -modelKwargs "{'hidden_size':1024, 'use_tp':True, 's2v':'lstm'}" -num_epochs 1000 -path2data ../dataset/kit-mocap -render_list subsets/render_list -s2v 1 -save_dir save/model -tb 1 -time 16 -transforms "['zNorm']"
```

This model has 2 training steps. `train_seq2seq.py` uses a seq2seq model to first learn an embedding for pose sequences. Once the training is complete, `train_wordConditioned.py` is called which optimizes to map from language embeddings to pose embeddings.

---
### Sampling

#### Sampling from trained Models
The training scripts will sample after the stopping criterion has reached, but if you would like to manually sample run the following script

```sh
python sample_wordConditioned.py -load <path-to-weights.p>
```

``<path-to-weights.p>`` ends in `_weights.p`

#### Using Pretrained Models

Make sure you have downloaded the pre-trained models as described [here](#pre-trained-models).
- JL2P

```sh
python sample_wordConditioned.py -load save/jl2p/exp_726_cpk_jointSampleStart_model_Seq2SeqConditioned9_time_16_chunks_1_weights.p
```

- Lin et. al. [1]
```sh
python sample_wordConditioned.py -load save/lin-et-al/exp_700_cpk_mooney_model_Seq2SeqConditioned10_time_16_chunks_1_weights.p 
```

---
### Rendering
After sampling, it would be nice to see what animation does the model generates. We only use the test samples for rendering.  

If possible, use a machine with many cpu cores, as rendering animations on matplotlib is painfully slow. `render.py` uses all the available cores for parallel processing.

#### Using your trained model
```sh
python render.py -dataset KITMocap -load <path-to-weights.p> -feats_kind fke -render_list subsets/render_list
```

#### Using pre-trained Models
- JL2P
```sh
python render.py -dataset KITMocap -load save/jl2p/exp_726_cpk_jointSampleStart_model_Seq2SeqConditioned9_time_16_chunks_1_weights.p -feats_kind fke -render_list subsets/render_list
```

- Lin et. al. [1]
```sh
python render.py -dataset KITMocap -load save/lin-et-al/exp_700_cpk_mooney_model_Seq2SeqConditioned10_time_16_chunks_1_weights.p -feats_kind fke -render_list subsets/render_list
```

### References
[1]: Lin, Angela S., et al. "1. Generating Animated Videos of Human Activities from Natural Language Descriptions." Learning 2018 (2018).
