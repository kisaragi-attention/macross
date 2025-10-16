## Paper Manuscript at [Here](./macross.pdf)

## How to Use

* Clone the project
* Create environment via conda

- Download the Miniconda installation script
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
- Run the installation script
```bash
sh Miniconda3-latest-Linux-x86_64.sh
```
- Check if the installation is successful
```bash
conda --version
```
- Create a virtual environment
```bash
conda create -n macross python=3.10.15
```
- Enter the virtual environment
```bash
conda activate macross
```
- Install dependencies
```bash
pip install -r macross_env.txt
```
- Test if the environment is successfully configured
```bash
python test_env.py
```

* Download the CIC EV charger attack dataset 2024 (CICEVSE2024) at [here](https://www.unb.ca/cic/datasets/evse-dataset-2024.html).

## Data Pre-processing

```bash
python preprocessing.py -s /path/to/CompletePcap/ -t processed_data
```

## Create Train and Test

```bash
python create_train_test_set.py -s processed_data -t train_test_data
```

## Train Model

Binary Classification

```bash
python train_macross.py -d train_test_data/binary_classification/train.parquet -m model/binary_classification.macross.model -t binary
```

Scenario Classification

```bash
python train_macross.py -d train_test_data/scenario_classification/train.parquet -m model/scenario_classification.macross.model -t scenario
```

Attack Classification

```bash
python train_macross.py -d train_test_data/attack_classification/train.parquet -m model/attack_classification.macross.model -t attack
```

## Evaluation Result

- Add Jupyter kernel
```bash
python -m ipykernel install --user --name macross --display-name macross
```
- Run Jupyter 
```bash
jupyter notebook
```
- Select an macross&ablation_evaluation.ipynb file and run it