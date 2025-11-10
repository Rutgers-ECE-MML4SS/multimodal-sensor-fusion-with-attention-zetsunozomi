# Instructions

##### 1. Setup Instructions

install via conda: 

```sh
conda env create -f environment.yml -n a2
conda activate a2
```

By default this repo runs with GPU. You will need to install pytorch with compatible CUDA version. Check with the following command:

```python
python3 -c "import torch; print(torch.cuda.is_available())"
```

When I setup this with CPU, extra pip3 install is required. Please install what you miss if error occurs.

##### 2. Training Command

Run：

```sh
sh train_early.sh
```

or any other .sh script with certain config.

Should you modify and config, please see the .yaml under the /config/ folder.

##### 3. Evaluation Command

Run:
```sh
sh eval.sh
```

##### 4. Uncertainty Analysis Command

Run:
```python
python3 src/visualize_attention.py
```
It will generate attention and uncertinty figure, and save uncertainty.json under experiments figure.

##### 5. Results Summary
We implemented modal fusion. Early and Late fusion strategy works best.
