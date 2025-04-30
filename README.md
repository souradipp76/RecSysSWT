# Sliding Window Training(SWT) for RecSys Foundation Models

This repository contains the code to reproduce the experiments of the paper "Is Sliding Window All You Need? Reproducing Industrial-Scale Long-Sequence Training for Recommender Systems".

## Installation

To create a virtual environment before installing, you can use the command:
```bash
conda create -n recsys python=3.11
conda activate recsys
pip install torch pandas numpy tqdm xxhash
```

## Dataset
Dowload the datasets using the following commands:
```bash
# Creat data directory
mkdir data
cd data

# Download RetailRocket dataset
curl -L -o ecommerce-dataset.zip https://www.kaggle.com/api/v1/datasets/download/retailrocket/ecommerce-dataset
unzip ecommerce-dataset.zip

# Dowload TaoBao dataset
curl -L -o userbehavior.zip https://www.kaggle.com/api/v1/datasets/download/marwa80/userbehavior
unzip userbehavior.zip
```

## Experiments

### Training
For running the experiments, navigate to the `src` directory.
```bash
cd ../src
```

To run the training and evaluation on RetailRocket dataset, use the follow command:
```bash
python sliding.py
```


To run the training and evaluation on TaoBao dataset, use the follow command:
```bash
python hashsliding.py
```

Make sure to set the correct mode (`control`, `sliding` or `mixed`) in the script along with the hyper-parameters `max_history` and `sliding_stride` in all the above cases based on the experiments.

### Evaluation
Additional Evaluation can be done using the following command:
```bash
python inference.py        # uses saved model.pth from training
```
Make sure to set the correct parameters similar to training.

## Citation

If you use this codebase in academic work, please cite:

```
@misc{recsysswt2025,
  title   = {RecSysSWT},
  author  = {Anonymous},
  year    = {2025},
  howpublished = {\url{https://github.com/anonymous/RecSysSWT}}
}
```

---

## License

Read the [LICENSE](LICENSE) file.
