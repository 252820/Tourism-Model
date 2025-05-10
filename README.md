# TSI
## 1.Environment Preparation
```
conda create -n tsi python=3.10
conda activate tsi
git clone https://github.com/anonymous-user.git
cd Tourism-Model
pip install -r requirements.txt
```

## 2.Preparing Pre-trained Checkpoints
Run the dnn-best-save.py to get pre-trained checkpoint for TSI.
```
python dnn/dnn-best-save.py
python model/TSI.py 
```

## 3.Warning
Synthetic data may overrepresent certain sarcasm patterns and should be handled with caution in real-world deployments.
