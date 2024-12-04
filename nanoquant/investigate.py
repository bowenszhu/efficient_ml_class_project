import torch
import tqdm
import os
import gc
from torch import nn
from transformers import GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pickle as pkl
# SmoothQuant
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import WQAQLinear, quantize_model
from smoothquant.calibration import get_act_scales
# AWQ
from huggingface_hub import hf_hub_download
from awq.quantize.pre_quant import apply_awq


class PerplexityEvaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        testenc = self.dataset
        nsamples = self.n_samples
        model = model.eval()

        nlls = []
        for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))

class AccuracyEvaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
    


class Investigation:
    def __init__(
            self,
            short_model_name: str = "opt-125m",
            repo_dir: str = ".",
            n_bits: int = 4, 
            q_group_size: int = 128,
            q_protect: bool = True,
            q_protection_ratio: float = 0.02,
            q_protection_scale: float = 2.0,
            q_smoothing_strength: float = 0.5,
            n_samples: int = 40,
        ):
        self.short_model_name = short_model_name
        if short_model_name.startswith("opt"):
            self.model_name = f"facebook/{short_model_name}"
        elif short_model_name.startswith("llama-2"):
            self.model_name = f"meta-llama/{short_model_name}-hf"
        elif short_model_name.startswith("llama-3"):
            self.model_name = f"meta-llama/{short_model_name}"
        else:
            raise ValueError("Unknown model name")
        # SmoothQuant act scales
        scales_path = f"{repo_dir}/act_scales/{short_model_name}.pt"
        assert os.path.exists(scales_path), f"Cannot find the act scales at {scales_path}"
        self.smooth_act_scales = torch.load(scales_path)
        # AWQ scales.
        assert short_model_name in ["opt-125m", "opt-6.7b", "opt-13b", "llama-2-7b"], "Only supported models for AWQ. Include more."
        if q_protect:
            awq_zoo = "mit-han-lab/awq-model-zoo"
            awq_pt_name = f"{short_model_name}-w4-g128.pt"
            awq_pt_filename = hf_hub_download(repo_id=awq_zoo, filename=awq_pt_name, repo_type="dataset")
            self.awq_pt = torch.load(awq_pt_filename, map_location="cpu")
        else:
            self.awq_pt = None
        # Make dataset.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=512)
        acc_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        perp_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        acc_dataset = load_dataset("lambada", split=f"validation[:{n_samples}]")
        self.perp_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.acc_evaluator = AccuracyEvaluator(acc_dataset, acc_tokenizer, self.device)
        self.perp_evaluator = PerplexityEvaluator(self.perp_dataset, perp_tokenizer, self.device, n_samples=n_samples)
        self.n_bits = n_bits
        self.q_group_size = q_group_size
        self.q_protect = q_protect
        self.q_protection_ratio = q_protection_ratio
        self.q_protection_scale = q_protection_scale
        self.q_smoothing_strength = q_smoothing_strength


    def make_base_model(self):
        print("Making base model...")
        model_fp16 = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16, device_map="auto"
        )
        print("Done making base model.")
        return model_fp16
    
    def evaluate_base_model(self, perp=True):
        model = self.make_base_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
    
    def make_base_awq_model(self):
        assert self.q_group_size == 128 # Only supported group size.
        assert self.q_protect, "Only supported for AWQ"
        model = self.make_base_model()
        print("Applying AWQ...")
        apply_awq(model, self.awq_pt)
        print("Done applying AWQ.")
        # Move to device
        model = model.to(self.device)
        return model
        

    def evaluate_base_awq_model(self, perp=True):
        model = self.make_base_awq_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
    

    
    def make_base_quantized_model(self):
        model_fp16 = self.make_base_model()
        print("Quantizing model...")
        q_model = quantize_model(
            model_fp16,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        print("Done quantizing model.")
        return q_model


    def evaluate_base_quantized_model(self, perp=True):
        model = self.make_base_quantized_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
        

    def make_base_smooth_model(self):
        model = self.make_base_model()
        print("Smoothing model...")
        smooth_lm(model, self.smooth_act_scales, self.q_smoothing_strength)
        print("Done smoothing model.")
        print("Quantizing model...")
        q_model = quantize_model(
            model,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        print("Done quantizing model.")
        return q_model
    
    def evaluate_base_smooth_model(
        self,
        perp=True,
    ):
        model = self.make_base_smooth_model()
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res
            

    def make_setup_model(self, apply_smooth=False):
        assert self.n_bits == 4
        assert self.q_group_size == 128
        if self.q_protect:
            model = self.make_base_awq_model()
        else:
            model = self.make_base_model()
        if apply_smooth:
            if self.q_protect:
                print("Computing scales after AWQ...")
                act_scales = get_act_scales(model, self.tokenizer, self.perp_dataset)
                print("Smoothing model...")
                smooth_lm(model, act_scales, self.q_smoothing_strength)
            else:
                print("Smoothing model...")
                smooth_lm(model, self.smooth_act_scales, self.q_smoothing_strength)
            print("Done smoothing model.")
        print("Quantizing model...")
        print(f"Quantizing model... {self.q_protect}")
        q_model = quantize_model(
            model,
            n_bits=self.n_bits,
            q_group_size=self.q_group_size,
            q_protect=self.q_protect,
            q_protection_ratio=self.q_protection_ratio,
            q_protection_scale=self.q_protection_scale,
        )
        print("Done quantizing model.")
        return q_model


    def evaluate_setup_model(self, apply_smooth=False, perp=True):
        model = self.make_setup_model(apply_smooth=apply_smooth)
        if perp:
            res = self.perp_evaluator.evaluate(model)
        else:
            res = self.acc_evaluator.evaluate(model)
        del model
        return res

        



def make_setup(n_bits, q_group_size, q_protect, q_protection_scale, q_protection_ratio, q_smoothing_strength):
    return {
        "n_bits": n_bits,
        "q_group_size": q_group_size,
        "q_protect": q_protect,
        "q_protection_scale": q_protection_scale,
        "q_protection_ratio": q_protection_ratio,
        "q_smoothing_strength": q_smoothing_strength,       
    }

def setup_name(setup):
    n_bits = setup["n_bits"]
    base_name = f"W{n_bits}A{n_bits}"
    q_group_size = setup["q_group_size"]
    if q_group_size > 0:
        base_name += f" G{q_group_size}"
    q_protect = setup["q_protect"]
    if q_protect:
        q_protection_scale = setup["q_protection_scale"]
        q_protection_ratio = setup["q_protection_ratio"]
        with_act = "Act" if q_protection_ratio > 1e-5 else "NoAct"
        if q_protection_scale > 1e-5:
            base_name += f" AWQ-Scaled-{with_act}"
        else:
            base_name += f" AWQ-Mixed-{with_act}"
    return base_name

def make_baselines():
    return ["fp16", "awq", "smoothquant", "smoothquant-g", "w4a4", "smooth-w4a4", "w8a8"]


def make_setups():
    q_group_size = 128
    n_bits = 4
    setups = []
    q_smoothing_strength = 0.5
    # No weight or activation protection
    q_protect = False
    setups.append(make_setup(n_bits, q_group_size, q_protect, -1.0, -1.0, q_smoothing_strength))
    # With protection
    q_protect = True
    # Weight-only protection.
    q_protection_scale = 0.0
    q_protection_ratio = 0.0
    setups.append(make_setup(n_bits, q_group_size, q_protect, q_protection_scale, q_protection_ratio, q_smoothing_strength))
    # # Mixed-precision activation protection
    # q_protection_scale = 0.0
    # q_protection_ratio = 0.03
    # setups.append(make_setup(n_bits, q_group_size, q_protect, q_protection_scale, q_protection_ratio, q_smoothing_strength))
    return setups


def sweep(short_model_name, repo_dir, save_dir, perp=True):
    os.makedirs(save_dir, exist_ok=True)
    result_file = f"{save_dir}/results_{short_model_name}.pkl"
    if os.path.exists(result_file):
        with open(result_file, "rb") as f:
            results = pkl.load(f)
    else:
        results = {}
    baselines = make_baselines()
    for baseline in baselines:
        if baseline in results:
            print(f"Baseline {baseline} already run. Results: {results[baseline]}")
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Running baseline {baseline}")
        if baseline == "fp16":
            investigation = Investigation(short_model_name, repo_dir)
            res = investigation.evaluate_base_model(perp=perp)
        elif baseline == "awq":
            investigation = Investigation(short_model_name, repo_dir, q_group_size=128)
            res = investigation.evaluate_base_awq_model(perp=perp)
        elif baseline == "smoothquant":
            investigation = Investigation(short_model_name, repo_dir, n_bits=8, q_group_size=0, q_protect=False)
            res = investigation.evaluate_base_smooth_model(perp=perp)
        elif baseline == "smoothquant-g":
            investigation = Investigation(short_model_name, repo_dir, n_bits=8, q_group_size=128, q_protect=False)
            res = investigation.evaluate_base_smooth_model(perp=perp)
        elif baseline == "w8a8":
            investigation = Investigation(short_model_name, repo_dir, n_bits=8, q_group_size=0, q_protect=False)
            res = investigation.evaluate_base_quantized_model(perp=perp)
        elif baseline == "w4a4":
            investigation = Investigation(short_model_name, repo_dir, n_bits=4, q_group_size=0, q_protect=False)
            res = investigation.evaluate_base_quantized_model(perp=perp)
        elif baseline == "smooth-w4a4":
            investigation = Investigation(short_model_name, repo_dir, n_bits=4, q_group_size=0, q_protect=False)
            res = investigation.evaluate_base_smooth_model(perp=perp)
        results[baseline] = res
        print(f"{baseline}: {res}")
        with open(result_file, "wb") as f:
            pkl.dump(results, f)
    setups = make_setups()
    for setup in setups:
        setup_key = str(setup)
        base_expt_name = setup_name(setup)
        if setup_key in results and base_expt_name != "W4A4 G128":
            print(f"Setup {base_expt_name} already run. Results={results[setup_key]['q_res']}, SmoothResults={results[setup_key]['q_smooth_res']}")
            continue
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        investigation = Investigation(short_model_name, repo_dir, **setup)
        simple_expt_name = f"{base_expt_name}"
        if simple_expt_name not in results:
            print(f"Running setup {base_expt_name}")
            q_res = investigation.evaluate_setup_model(perp=perp, apply_smooth=False)
            results[simple_expt_name] = q_res
            with open(result_file, "wb") as f:
                pkl.dump(results, f)
        else:
            q_res = results[simple_expt_name]
        print(f"{simple_expt_name}: {q_res}")
        # Smoothed model
        smooth_expt_name = f"Smooth {base_expt_name}"
        if smooth_expt_name not in results:
            print(f"Running setup {smooth_expt_name}")
            q_smooth_res = investigation.evaluate_setup_model(perp=perp, apply_smooth=True)
            results[smooth_expt_name] = q_smooth_res
            with open(result_file, "wb") as f:
                pkl.dump(results, f)
        else:
            q_smooth_res = results[smooth_expt_name]
        print(f"{smooth_expt_name}: {q_smooth_res}")
        res = {
            "setup": setup,
            "q_res": q_res,
            "q_smooth_res": q_smooth_res,
        }
        results[setup_key] = res
        # Checkpointing
        with open(result_file, "wb") as f:
            pkl.dump(results, f)




def report_sweep(short_model_name, save_dir):
    result_file = f"{save_dir}/results_{short_model_name}.pkl"
    with open(result_file, "rb") as f:
        results = pkl.load(f)

    baselines = make_baselines()
    for baseline in baselines:
        base_result = results[baseline]
        print(f"{baseline},{base_result}")
    setups = make_setups()
    for setup in setups:
        setup_key = str(setup)
        res = results[setup_key]
        setup = res["setup"]
        base_expt_name = setup_name(setup)
        simple_expt_name = f"{base_expt_name}"
        smooth_expt_name = f"Smooth {base_expt_name}"
        print(f"{simple_expt_name},{res['q_res']}")
        print(f"{smooth_expt_name},{res['q_smooth_res']}")
