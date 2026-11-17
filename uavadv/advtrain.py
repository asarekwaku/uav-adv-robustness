"""Adversarial augmentation / training (proposal Sec. IV-C, stronger baseline).

A DetectionTrainer whose preprocess_batch replaces a fraction of every batch with adversarial
copies generated on the fly against the current weights (Madry et al. 2018; fast FGSM with random
start from Wong et al. 2020). Gradients are taken with torch.autograd.grad w.r.t. the input only, so
parameter .grad buffers are not touched by the attack.
"""
from __future__ import annotations

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model


def make_adv_trainer(attack: str, eps255: float, ratio: float, pgd_steps: int = 3):
    eps = eps255 / 255

    class AdvTrainer(DetectionTrainer):
        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            n = int(round(batch["img"].shape[0] * ratio))
            if n == 0:
                return batch
            model = unwrap_model(self.model)
            was_training = model.training
            model.eval()  # freeze BN statistics while crafting the perturbation
            x = batch["img"][:n].detach().float()
            sel = batch["batch_idx"] < n
            sub = {"img": x, "batch_idx": batch["batch_idx"][sel], "cls": batch["cls"][sel], "bboxes": batch["bboxes"][sel]}

            def grad(z):
                z = z.detach().requires_grad_(True)
                loss, _ = model.loss({**sub, "img": z}, model(z))
                return torch.autograd.grad(loss.sum(), z)[0]

            with torch.autocast(device_type=self.device.type, enabled=False):
                if attack == "fgsm_rs":
                    delta = (torch.rand_like(x) * 2 - 1) * eps
                    z = (x + delta).clamp(0, 1)
                    z = z + 1.25 * eps * grad(z).sign()
                elif attack == "pgd":
                    z = (x + (torch.rand_like(x) * 2 - 1) * eps).clamp(0, 1)
                    for _ in range(pgd_steps):
                        z = z + 2.5 * eps / pgd_steps * grad(z).sign()
                        z = torch.min(torch.max(z, x - eps), x + eps).clamp(0, 1)
                else:
                    raise ValueError(attack)
                z = torch.min(torch.max(z, x - eps), x + eps).clamp(0, 1)
            model.train(was_training)
            batch["img"] = torch.cat([z.to(batch["img"].dtype), batch["img"][n:]], 0).detach()
            return batch

    return AdvTrainer
