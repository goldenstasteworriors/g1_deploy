"""OpenWAM/Accelerate training with auditable conditioning and held-out loss."""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from train import _build_accelerator
from lab_low.dataset import LabDataset
from lab_low.skills import ACTION_DIM, STATE_DIM, SKILLS, HANDS
from openwam.train.openwam_trainer import OpenWAMTrainer
from openwam.train.utils.seeding import seed_everything


class LabTrainer(OpenWAMTrainer):
    def build_dataloader(self, batch_size):
        if not self.cfg.dataloader.get("balance_skills", False):
            return super().build_dataloader(batch_size)
        from openwam.train.utils.seeding import make_dataloader_generator, dataloader_worker_init_fn
        skills = [self.dataset.segments[segment]["skill"] for segment, _ in self.dataset.windows]
        counts = Counter(skills)
        weights = torch.tensor([1. / counts[skill] for skill in skills], dtype=torch.double)
        seed = int(self._run_seed) if self._run_seed is not None else 42
        generator = make_dataloader_generator(seed, rank=self._rank)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        print("SKILL_SAMPLING " + json.dumps(dict(windows=dict(counts),
              expected_probability={skill: 1./len(counts) for skill in counts})), flush=True)
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, sampler=sampler,
            num_workers=int(self.cfg.training.dataset_num_workers), collate_fn=list, pin_memory=True,
            generator=generator, worker_init_fn=dataloader_worker_init_fn)

    def __init__(self, cfg, accelerator, dataset):
        super().__init__(cfg, accelerator, dataset)
        self.validation = LabDataset(cfg.dataloader, "val")
        self.gradient_checks = {}
        self.best_score = float("inf")
        self.required_gradients = {
            "lab_conditioner.skill.weight", "lab_conditioner.hand.weight",
            "lab_conditioner.query", "lab_conditioner.visual.weight",
            "lab_conditioner.state_action.weight", "action_backbone.action_decoder.weight",
        }
        for i, block in enumerate(self.architecture.lab_conditioner.visual_memory.blocks):
            self.required_gradients.add(f"lab_conditioner.visual_memory.blocks.{i}.spatial.in_proj_weight")
            if block.temporal is not None:
                self.required_gradients.add(f"lab_conditioner.visual_memory.blocks.{i}.temporal.in_proj_weight")
        if cfg.verify_gradients:
            arch = self.architecture
            for name, param in arch.named_parameters():
                if name in self.required_gradients:
                    param.register_hook(self.grad_hook(name))
            # These are separate projected contexts. Checking appended tokens in
            # BOTH proves each expert consumed the new conditions in the real MoT.
            for name, module in (("video_context", arch.video_backbone.dit.text_embedding),
                                 ("action_context", arch.action_backbone.text_embedding)):
                def hook(module, args, output, key=name):
                    if output.requires_grad:
                        output.register_hook(self.grad_hook(key, context=True))
                module.register_forward_hook(hook)

    def grad_hook(self, name, context=False):
        def check(grad):
            if context:
                count = self.architecture.last_lab_token_count
                if count < 1:
                    raise RuntimeError("Missing lab context token count")
                grad = grad[:, -(count+1):-1]  # lab tokens immediately before proprio
            self.gradient_checks[name] = dict(finite=bool(torch.isfinite(grad).all()),
                nonzero=bool(torch.count_nonzero(grad)), norm=float(grad.float().norm()))
        return check

    def setup_output_dir(self, *args, **kwargs):
        output, resume = super().setup_output_dir(*args, **kwargs)
        self.run_output = Path(output)
        if self.accelerator.is_main_process:
            (self.run_output/"transfer_report.json").write_text(json.dumps(self.architecture.transfer_report, indent=2))
            (self.run_output/"data_contract.json").write_text(json.dumps(dict(
                train_episodes=self.dataset.train_ids, val_episodes=self.dataset.val_ids,
                excluded=self.dataset.excluded, train_windows=len(self.dataset), val_windows=len(self.validation),
                target_mode="initial_reference_or_unmarked_held_dispense", future_tracking_annotations=False,
                unmarked_train_segments=sum(not s["marks"] for s in self.dataset.segments),
                unmarked_val_segments=sum(not s["marks"] for s in self.validation.segments),
                action_source="observation.state", action_alignment="state[t] -> state[t+1:t+33]",
                memory_motion="past achieved observation delta, NOT executed command",
                memory_architecture=self.architecture.conditioner_version,
                visual_memory="causal spatial/temporal patches before target pooling; reference independent",
                state_memory="one continuous token per rolling observation; invalid tokens masked",
                completion_supervision=False,
                jitter_source=self.cfg.dataloader.jitter_observation_key or "root_orientation angular-speed proxy, rad/s",
                action_dim=ACTION_DIM, state_dim=STATE_DIM, manipulation_only=True,
                skill_vocabulary=list(SKILLS), hand_ids=HANDS,
                cameras=3, history_offsets=list(self.cfg.dataloader.history_offsets),
                augmentation={key: OmegaConf.to_container(self.cfg.dataloader[key], resolve=True)
                    for key in ("photometric", "box_augmentation", "camera_augmentation")
                    if key in self.cfg.dataloader},
                balanced_skill_sampling=bool(self.cfg.dataloader.get("balance_skills", False)),
                skills=sorted(set(s["skill"] for s in self.dataset.segments)),
                hands=sorted(set(s["hand"] for s in self.dataset.segments))), indent=2))
        print("RUN_OUTPUT="+str(output), flush=True)
        return output, resume

    @torch.no_grad()
    def validate(self, step):
        arch = self.accelerator.unwrap_model(self.architecture)
        was_training = arch.training
        py, npstate = random.getstate(), np.random.get_state()
        values = []
        try:
            arch.eval()
            # Cover each represented skill; keep these windows fixed across evaluations.
            by_skill = {}
            for index, (segment, _) in enumerate(self.validation.windows):
                skill = self.validation.segments[segment]["skill"]
                by_skill.setdefault(skill, []).append(index)
            quota = max(1, int(self.cfg.validation_windows)//len(by_skill))
            indices = [pool[i] for _, pool in sorted(by_skill.items())
                       for i in np.linspace(0,len(pool)-1,min(quota,len(pool)),dtype=int)]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for i, index in enumerate(indices):
                    torch.manual_seed(73000+i)
                    result = self.compute_loss([self.validation[int(index)]])
                    values.append({k: float(v) for k, v in result.items()})
        finally:
            arch.train(was_training)
            random.setstate(py)
            np.random.set_state(npstate)
        scores = {k: float(np.mean([r[k] for r in values])) for k in values[0]}
        if not all(np.isfinite(v) for v in scores.values()):
            raise FloatingPointError(scores)
        record = dict(step=step, **scores, windows=len(values), metric="held-out flow loss, not rollout success")
        if self.accelerator.is_main_process:
            with (self.run_output/"validation.jsonl").open("a") as f:
                f.write(json.dumps(record)+"\n")
            if scores["action"] < self.best_score:
                self.best_score = scores["action"]
                best = self.run_output/"checkpoint_best.safetensors"
                arch.save_checkpoint(best)
                (self.run_output/"best_metric.json").write_text(json.dumps(dict(record, checkpoint=str(best)), indent=2))
        print("VALIDATION "+json.dumps(record), flush=True)

    def log_step(self, **kwargs):
        super().log_step(**kwargs)
        arch = self.accelerator.unwrap_model(self.architecture)
        metrics = {k: float(v) for k, v in kwargs["metrics"].items()}
        metrics.update(step=kwargs["global_step"], optimizer_step=kwargs["opt_step"],
            jitter_weight=arch.last_jitter_weight, peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(metrics)
        if self.accelerator.is_main_process:
            with (self.run_output/"metrics.jsonl").open("a") as f:
                f.write(json.dumps(metrics)+"\n")
            (self.run_output/"gradient_checks.json").write_text(json.dumps(self.gradient_checks, indent=2))
        print("TRAIN "+json.dumps(metrics), flush=True)
        if self.cfg.verify_gradients and kwargs["global_step"] == min(3, int(self.cfg.training.max_steps or 3)):
            expected = self.required_gradients | {"video_context", "action_context"}
            if expected-set(self.gradient_checks) or not all(v["finite"] and v["nonzero"] for v in self.gradient_checks.values()):
                raise RuntimeError(f"Disconnected conditions: {self.gradient_checks}")
        every = int(self.cfg.validation_every)
        if every and kwargs["global_step"] % every == 0:
            self.validate(kwargs["global_step"])

    def finish_training(self, output_path, global_step, save_steps, keep_last_k, is_main, wandb_run):
        final = Path(output_path)/f"checkpoint_step_{global_step}.safetensors"
        super().finish_training(output_path, global_step, 0 if final.exists() else save_steps,
                                keep_last_k, is_main, wandb_run)
        if is_main and final.exists():
            arch = self.accelerator.unwrap_model(self.architecture)
            # Every stored tensor is loaded from disk and compared with live
            # weights without allocating a second 5B backbone on the GPU.
            current = arch.state_dict()
            with safe_open(str(final), framework="pt", device="cpu") as f:
                count = 0
                for key in f.keys():
                    if not torch.equal(f.get_tensor(key), current[key].detach().cpu()):
                        raise RuntimeError(f"Checkpoint readback mismatch: {key}")
                    count += 1
            (Path(output_path)/"checkpoint_readback.json").write_text(json.dumps(dict(
                checkpoint=str(final), tensors=count, exact_match=True,
                scope="all saved delta tensors; frozen video weights supplied by foundation"), indent=2))
            metric_path = Path(output_path)/"best_metric.json"
            if metric_path.exists():
                metric = json.loads(metric_path.read_text())
                if metric["step"] == global_step:
                    duplicate = Path(output_path)/"checkpoint_best.safetensors"
                    if duplicate.exists():
                        duplicate.unlink()  # own duplicate, only after exact readback
                    metric["checkpoint"] = str(final)
                    metric_path.write_text(json.dumps(metric, indent=2))


@hydra.main(version_base=None, config_path=str(ROOT/"configs"), config_name="lab")
def main(cfg):
    # This validated launcher is deliberately single-GPU. Do not silently
    # duplicate best checkpoint writes in a distributed run.
    import os
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one GPU for the validated lab launcher")
    seed_everything(int(cfg.project.seed), rank=0)
    accelerator = _build_accelerator(cfg)
    dataset = LabDataset(cfg.dataloader)
    trainer = LabTrainer(cfg, accelerator, dataset)
    trainer.train()
    print("COMPLETED "+str(trainer.run_output), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
