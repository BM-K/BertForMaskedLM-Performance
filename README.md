# BertForMaskedLM
- [BERT](https://arxiv.org/abs/1810.04805)의 Masked Language Model(MLM) 성능 체크

### 특정 데이터셋에 대한 Fine-tuning 없이 MLM 성능 체크
- Base Dataset: IMDB <br>
    - test set: 25,000개로 실험
```
python train.py --MLM True --Finetune False --max_len 512 --masking_ratio 0.15
```
- Result
    - *MLM Acc: **40.07%***

### 특정 데이터셋에 대한 Fine-tuning 후 MLM 성능 체크
- Base Dataset: IMDB <br>
    - train set: 20,000
    - valid set: 5,000
    - test set: 25,000
```
# Fist step: Fine-tuning (Sentiment Classification)
python train.py --MLM False --Finetune False --max_len 512 --batch_size 16 --masking_ratio 0.15 --train True --test True

# Second step: MLM
python train.py --MLM True --Finetune True --max_len 512 --batch_size 16 --masking_ratio 0.15
```
- Results
    - *Classification Acc: **93.46%***
    - *MLM Acc: **36.72%***
