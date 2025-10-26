from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data import OnlyXDataset
from src.estimation.utils import calculate_roughness


class BaseEstimator(ABC):
    def __init__(self) -> None:
        self.metrics_names = []

    @abstractmethod
    def estimate(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        return NotImplementedError("This method should be implemented by subclasses.")

    def get_metrics_names(self) -> List[str]:
        return self.metrics_names


class ClassifierEstimator(BaseEstimator):
    def __init__(self, n_classes: int) -> None:

        average = 'macro' if n_classes > 2 else 'binary'

        self.sklearn_metrics = {
            "accuracy": (accuracy_score, {}),
            "precision": (precision_score, {"average": average}),
            "recall": (recall_score, {"average": average}),
            "f1": (f1_score, {"average": average}),
        }
        self.metrics_names = list(self.sklearn_metrics.keys()) + [
            "balance_true",
            "balance_pred",
            "certainty",
        ]

    @staticmethod
    def balance(y: np.ndarray):
        return np.mean(y).item()

    @staticmethod
    def uncertainty(y_pred_probs: np.ndarray) -> float:
        prob_1 = y_pred_probs
        prob_0 = 1 - prob_1
        max_prob = np.max(np.stack([prob_1, prob_0], axis=1), axis=1)

        assert max_prob.shape == y_pred_probs.shape

        return np.mean(max_prob).item()

    def estimate(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_probs: np.ndarray
    ) -> List[float]:
        metrics_res = []
        for _, (metric_func, params) in self.sklearn_metrics.items():
            metrics_res.append(metric_func(y_true, y_pred, **params))

        metrics_res.append(self.balance(y_true))
        metrics_res.append(self.balance(y_pred))
        metrics_res.append(self.uncertainty(y_pred_probs))
        return metrics_res


class AttackEstimator(BaseEstimator):
    def __init__(
        self,
        disc_models: List[torch.nn.Module] = None,
        metric_effect: str = "F1",
        metric_hid: str = "PROB_HID",
        batch_size: int = None,
        n_classes: int = 2,
    ) -> None:
        
        multi_class_params = {'multi_class':'ovr'} if n_classes > 2 else {}
        average = 'macro' if n_classes > 2 else 'binary'

        self.metrics = {
            "ACC": (accuracy_score,{}),
            "ROC": (roc_auc_score, multi_class_params),
            "PR": (average_precision_score, {}),
            "F1": (f1_score,{"average": average}),
        }

        self.metric_effect = metric_effect
        self.metric_hid = metric_hid
        self.batch_size = batch_size
        self.n_classes = n_classes

        self.metrics_names = list(self.metrics.keys()) + ["EFF", "FR", "L1", "ACC_CORRECT", "ACC_ORIG_ADV", "ROUGHNESS", "ROUGHNESS_NORM"]

        self.calculate_hid = bool(disc_models)
        if disc_models:
            self.disc_models = disc_models
            self.metrics_names += [
                "PROB_HID",
                "ACC_DISC",
                "F1_DISC",
                "ROC_AUC_DISC",
                "CONC",
                "F_EFF_CONC",
            ]

    @staticmethod
    def accuracy_correct_predicted(y_true: np.ndarray, y_pred: np.ndarray, y_pred_orig: np.ndarray):
        orig_correct_mask = y_pred_orig == y_true
        return accuracy_score(y_true[orig_correct_mask], y_pred[orig_correct_mask])


    def calculate_effectiveness(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        y_pred_classes: np.ndarray
    ) -> Dict[str, float]:
        metric_res = {}

        for metric_name, (metric_func, params) in self.metrics.items():
            try:
                if metric_name in ["ROC", "PR"]:  # needs probabilities
                    metric_res[metric_name] = metric_func(y_true, y_pred_probs, **params)
                else:  # works with class labels
                    metric_res[metric_name] = metric_func(y_true, y_pred_classes, **params)
            except Exception as e:
                print(f"Failed to compute {metric_name}: {e}")
                metric_res[metric_name] = np.nan

        metric_res["EFF"] = 1 - metric_res[self.metric_effect]
        return metric_res

    def calculate_hid_one_model(
        self, X: torch.Tensor, disc_model: torch.nn.Module
    ) -> torch.Tensor:
        device = next(disc_model.parameters()).device
        if self.batch_size:
            loader = DataLoader(
                OnlyXDataset(X), batch_size=self.batch_size, shuffle=False
            )
            y_all_preds = torch.tensor([], device=device)
            for X_batch in loader:
                y_pred = disc_model(X_batch)
                y_all_preds = torch.concat([y_all_preds, y_pred], axis=0)
            return y_all_preds
        else:
            y_pred = disc_model(X)
            return y_pred

    def calculate_hiddeness(self, X_orig, X_adv: np.ndarray, step_id: int) -> Dict[str, float]:

        model_device = next(self.disc_models[0].parameters()).device
        X_adv = torch.tensor(X_adv).to(model_device)
        X_orig = torch.tensor(X_orig).to(model_device)

        y_pred_orig_prob = np.array([])
        y_pred_adv_prob = np.array([])
        n_objects = X_orig.shape[0]
        with torch.no_grad():
            for disc_model in self.disc_models:
                disc_predictions = (
                    self.calculate_hid_one_model(X_orig, disc_model)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                y_pred_orig_prob = np.concatenate([y_pred_orig_prob, disc_predictions])

                disc_predictions = (
                    self.calculate_hid_one_model(X_adv, disc_model)
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )
                y_pred_adv_prob = np.concatenate([y_pred_adv_prob, disc_predictions])

        y_pred_orig_prob = y_pred_orig_prob.reshape((n_objects, len(self.disc_models)))
        y_pred_adv_prob = y_pred_adv_prob.reshape((n_objects, len(self.disc_models)))
        assert y_pred_adv_prob.shape == (n_objects, len(self.disc_models))

        prob_hid = np.max(np.mean(y_pred_adv_prob, axis=0))

        y_pred_orig_prob = np.max(y_pred_orig_prob, axis=1)
        y_pred_adv_prob = np.min(y_pred_adv_prob, axis=1)

        assert len(y_pred_adv_prob) == len(X_adv)

        y_pred_prob = np.concatenate([y_pred_orig_prob, y_pred_adv_prob], axis=0)
        y_pred = np.round(y_pred_prob)

        if step_id > 0:
            y_true = np.concatenate(
                [np.zeros(y_pred_orig_prob.shape), np.ones(y_pred_adv_prob.shape)], axis=0
            )
        else:
            y_true = np.concatenate(
                [np.zeros(y_pred_orig_prob.shape), np.zeros(y_pred_adv_prob.shape)], axis=0
            )

        acc_disc = accuracy_score(y_true, y_pred)
        f1_disc = f1_score(y_true, y_pred)
        roc_auc_disc = roc_auc_score(y_true, y_pred_prob) if step_id > 0 else 1.0

        results = {
            "PROB_HID": prob_hid,
            "ACC_DISC": acc_disc,
            "F1_DISC": f1_disc,
            "ROC_AUC_DISC": roc_auc_disc,
        }
        results["CONC"] = 1 - results[self.metric_hid]
        return results

    @staticmethod
    def calculate_l1(X_orig: np.ndarray, X_adv: np.ndarray) -> float:

        # print("sdjhfljlksjfkjsklfjklf", X_orig.shape, X_adv.shape)
        # X_orig = X_orig.squeeze(-1)
        # X_adv = X_adv.squeeze(-1)

        data_shape_no_ax0 = tuple(np.arange(1, len(X_adv.shape)))
        l1_vector = np.sum(np.abs(X_orig - X_adv), axis=data_shape_no_ax0)
        assert l1_vector.shape[0] == len(X_orig)
        l1 = np.mean(l1_vector)
        return l1.item()

    @staticmethod
    def calculate_f_eff_conc(effectiveness: float, concealability: float) -> float:
        num = 2 * effectiveness * concealability
        denum = (effectiveness + concealability)
        return  num / denum if denum != 0 else 0

    def estimate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_classes: np.ndarray,
        y_pred_orig: np.ndarray,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
        step_id: int,
    ) -> List[float]:
        # assert y_true.shape == y_pred.shape
        # assert X_orig.shape == X_adv.shape

        # print(y_pred.shape)

        metrics = self.calculate_effectiveness(y_true, y_pred, y_pred_classes)

        metrics["L1"] = self.calculate_l1(X_orig, X_adv)
        metrics["ACC_ORIG_ADV"] = accuracy_score(y_pred_orig, y_pred_classes)
        metrics["FR"] = 1 - metrics["ACC_ORIG_ADV"]
        metrics['ACC_CORRECT'] =  self.accuracy_correct_predicted(y_true, y_pred_classes, y_pred_orig)
        metrics['ROUGHNESS'] = calculate_roughness(X_adv)
        metrics['ROUGHNESS_NORM'] = metrics['ROUGHNESS']/calculate_roughness(X_orig)

        if self.calculate_hid:
            metric_hid = self.calculate_hiddeness(X_orig, X_adv, step_id)
            metric_hid["F_EFF_CONC"] = self.calculate_f_eff_conc(
                metrics["EFF"], metric_hid["CONC"]
            )
            metrics.update(metric_hid)

        return_order_metrics = [metrics[name] for name in self.metrics_names]
        return return_order_metrics
    