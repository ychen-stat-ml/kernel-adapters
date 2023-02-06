# kernel-adapters
This is the official implementation for "Inducer-tuning: Connecting Prefix-tuning and Adapter-tuning" (EMNLP 2022) and "Empowering Parameter-Efficient Transfer Learning by Recognizing the Kernel Structure in Attention" (NAACL 2022 Findings).

## Installation

To prepare the conda environment for the code in this repo, the users can create the environment through
```sh
conda env create -f adapter.yml
```
which mainly involves the following package:
```sh
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c pytorch ignite=0.4.5
conda install -c "conda-forge/label/cf202003" tensorboard=2.5.0
conda install -c "conda-forge/label/cf202003" tensorflow=2.5.2
conda install -c "conda-forge/label/cf202003" transformers=4.10.2
conda install -c "conda-forge/label/cf202003" progress
conda install -c anaconda nltk
conda install scikit-image
conda install unidecode
conda install natsort
pip install datasets==1.16.1
conda install -c anaconda scikit-learn
```
We also need to run the following Python code before the NLG experiments.
```python
import nltk
nltk.download('punkt')
```

## NLG tasks using GPT-2

The code for nlg tasks is partially adapted from [this repo](https://github.com/zlinao/VGLM).

### Setup Data

- Download `data.zip` from [here](https://1drv.ms/u/s!AqsZ7ICy6kBI41-2ALmXJh3qV_l_?e=cjaJhJ), and unzip it. There will be a `data` folder.
- Place the `data` folder under the directory `kernel-adapters/nlg/`.

### Run Code

The initialization directory is the root directory `./kernel-adapters`.

```
cd nlg
sh scripts/run_nlg.sh
```

## NLU tasks using RoBERTa

The code for nlu tasks is partially adapted from [this repo](https://github.com/jxhe/unify-parameter-efficient-tuning).

### Run Code

The initialization directory is the root directory `./kernel-adapters`.

```
cd nlu
sh scripts/run_glue.sh
```

## Citation

If you find the repository helpful, please consider citing our papers:

```
@inproceedings{chen-etal-2022-inducer,
    title = "Inducer-tuning: Connecting Prefix-tuning and Adapter-tuning",
    author = "Chen, Yifan  and
      Hazarika, Devamanyu  and
      Namazifar, Mahdi  and
      Liu, Yang  and
      Jin, Di  and
      Hakkani-Tur, Dilek",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```

```
@inproceedings{chen-etal-2022-empowering,
    title = "Empowering parameter-efficient transfer learning by recognizing the kernel structure in self-attention",
    author = "Chen, Yifan  and
      Hazarika, Devamanyu  and
      Namazifar, Mahdi  and
      Liu, Yang  and
      Jin, Di  and
      Hakkani-Tur, Dilek",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
