from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
from torch.nn import functional as F

class CustomMetrics:
    def __init__(self):
        pass

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # Преобразование логитов в токены
        predictions = np.argmax(predictions, axis=-1)
        
        # Маскируем padding токены
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        
        # 1. Token Accuracy - базовая метрика точности
        token_acc = accuracy_score(labels, predictions)

        # 2. Precision, Recall, F1 - комплексные метрики качества
        precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                                 predictions, 
                                                                 average='macro', 
                                                                 zero_division=0)

        # 3. Cross-entropy loss - основная метрика потерь
        try:
            ce_loss = F.cross_entropy(torch.tensor(predictions).float(), 
                                    torch.tensor(labels).long())
        except:
            ce_loss = torch.tensor(float('inf'))

        return {
            "eval_token_accuracy": float(token_acc),
            "eval_precision": float(precision),
            "eval_recall": float(recall),
            "eval_f1": float(f1),
            "eval_ce_loss": float(ce_loss)
        }