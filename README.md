# DistilBERT IMDB — Knowledge Distillation

## About
Knowledge Distillation from BERT (Teacher) → DistilBERT (Student)
for IMDB sentiment analysis.

## Models on HuggingFace
- Teacher: https://huggingface.co/Sachin-Rathore-1234/bert-imdb-sentiment
- Student: https://huggingface.co/Sachin-Rathore-1234/distilbert-imdb-distilled

## Results
| Metric         | Teacher (BERT) | Student (DistilBERT) |
|----------------|----------------|----------------------|
| Accuracy       | ~91%           | ~89%                 |
| Parameters     | 110M           | 66M                  |
| Size Reduction | -              | 40% smaller          |
| Speed          | 1x             | 1.6x faster          |

## Distillation Settings
| Setting     | Value |
|-------------|-------|
| Temperature | 4.0   |
| Alpha       | 0.7   |
| Epochs      | 3     |
| Dataset     | IMDB  |

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Inference
```python
from transformers import pipeline

classifier = pipeline(
    'sentiment-analysis',
    model='Sachin-Rathore-1234/distilbert-imdb-distilled'
)
print(classifier("This movie was amazing!"))
```
