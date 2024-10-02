import numpy as np
from typing import List, Dict
import os
import json
from sklearn.metrics import roc_auc_score
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
import argparse
from pprint import pprint
from loguru import logger

class Evaluator:
    def __init__(self):
        self.results = []
        self.labels = []

    def add_result(self, result: Dict, label: Dict):
        """Add a single scanning result to the evaluator."""
        self.results.append(result)
        self.labels.append(label)

    def compute_accuracy(self) -> float:
        """Compute the overall accuracy of the scanning results."""
        if not self.results:
            return 0.0
        correct = sum(1 for result, label in zip(self.results, self.labels)
                      if result['is_backdoor'] == label['is_backdoor'])
        return correct / len(self.results)

    def compute_precision(self) -> float:
        """Compute the precision of the scanning results."""
        if not self.results:
            return 0.0
        true_positives = sum(1 for result, label in zip(self.results, self.labels)
                              if result['is_backdoor'] and label['is_backdoor'])
        false_positives = sum(1 for result, label in zip(self.results, self.labels)
                               if result['is_backdoor'] and not label['is_backdoor'])
        return true_positives / (true_positives + false_positives)

    def compute_recall(self) -> float:
        """Compute the recall of the scanning results."""
        if not self.results:
            return 0.0
        true_positives = sum(1 for result, label in zip(self.results, self.labels)
                              if result['is_backdoor'] and label['is_backdoor'])
        false_negatives = sum(1 for result, label in zip(self.results, self.labels)
                               if not result['is_backdoor'] and label['is_backdoor'])
        return true_positives / (true_positives + false_negatives)

    def compute_f1_score(self) -> float:
        """Compute the F1 score of the scanning results."""
        if not self.results:
            return 0.0
        precision = self.compute_precision()
        recall = self.compute_recall()
        return 2 * (precision * recall) / (precision + recall)

    def compute_roc_auc_score(self) -> float:
        """Compute the ROC AUC score of the scanning results."""
        if not self.results:
            return 0.0
        
        y_true = [label['is_backdoor'] for label in self.labels]
        y_scores = [result['is_backdoor'] for result in self.results]
        
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.0

    def compute_bleu_score(self) -> float:
        """Compute the average BLEU score of the scanning results for backdoor samples."""
        if not self.results or 'target' not in self.results[0]:
            return 0.0
        
        bleu_scores = []
        for result, label in zip(self.results, self.labels):
            if label['is_backdoor'] and 'target' in result and 'target' in label:
                reference = word_tokenize(label['target'].lower())
                candidate = word_tokenize(result['target'].lower())
                score = sentence_bleu([reference], candidate)
                bleu_scores.append(score)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

    def compute_overhead(self) -> float:
        """Compute the average overhead (time taken) of the scanning results."""
        if not self.results or 'overhead' not in self.results[0]:
            return 0.0
        
        total_overhead = sum(result['overhead'] for result in self.results if 'overhead' in result)
        return total_overhead / len(self.results)
     
    def generate_report(self) -> Dict:
        """Generate a comprehensive report of all metrics."""
        return {
            "accuracy": self.compute_accuracy(),
            "precision": self.compute_precision(),
            "recall": self.compute_recall(),
            "f1_score": self.compute_f1_score(),
            "roc_auc_score": self.compute_roc_auc_score(),
            "bleu_score": self.compute_bleu_score(),
            "overhead": self.compute_overhead()
        }
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="result/bait_cba")
    parser.add_argument("--output_dir", type=str, default="result/bait_cba")
    args = parser.parse_args()

    test_dir = args.test_dir
    output_dir = args.output_dir
    
    eval = Evaluator()
    for model_id in os.listdir(test_dir):
        if os.path.isdir(os.path.join(test_dir, model_id)):
            args = json.load(open(os.path.join(test_dir, model_id, "arguments.json"), "r"))
            label = {
                "is_backdoor": bool(args["model_args"]["is_backdoor"]),
                "target": args["model_args"]["target"]
            }
            
            output = json.load(open(os.path.join(test_dir, model_id, "result.json"), "r"))
            result = {
                "is_backdoor": bool(output["is_backdoor"]),
                "target": output["invert_target"],
                "overhead": output["time_taken"]
            }
            eval.add_result(result, label)
    
    logger.info(f"evaluating BAIT results for {len(eval.results)} models from {test_dir}...")
    metrics = eval.generate_report()
    logger.info(metrics)
    
    # Save to CSV
    with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
        f.write("Metric,Value\n")
        for metric, value in metrics.items():
            f.write(f"{metric.capitalize()},{value:.4f}\n")

    logger.info(f"Metrics saved to {output_dir}/metrics.csv")
