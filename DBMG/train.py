import os
import argparse

import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
import torch_npu
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from dataprocess import get_dataloaders
from encoder import DBMGEncoder
from lossf import AdaptiveLossWeighting, category_softmax_loss, pairwise_ranking_loss
from matcher import DBMG
from metric import compute_map


def init_distributed(args):
    torch.npu.set_compile_mode(jit_compile=True)
    torch.npu.set_float32_matmul_precision("high")

    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.npu.set_device(args.rank)
    args.device = torch.device(f"npu:{args.rank}")
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    return args


def build_dataloaders(args):
    return get_dataloaders(args)


def train_one_epoch(epoch, encoder, matcher, train_loader, optimizer, loss_fct, adaptive_loss, args):
    encoder.train()
    matcher.train()
    total_loss, best_map = 0.0, 0.0

    if hasattr(train_loader.sampler, "set_epoch"):
        train_loader.sampler.set_epoch(epoch)

    for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=args.rank != 0):
        for key in batch:
            batch[key] = batch[key].to(args.device)

        features = encoder(
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            pixel_values=batch["pixel_values"],
            image_description_input_ids=batch["image_description_input_ids"],
            image_description_attention_mask=batch["image_description_attention_mask"],
        )

        logits = matcher(
            features["image_description_embeds"],
            features["image_description_seq_tokens"],
            features["text_embeds"],
            features["text_seq_tokens"],
            features["image_embeds"],
            features["image_patch_tokens"],
        )

        labels = torch.arange(len(features["text_embeds"]), device=args.device).long()
        loss_i2t = loss_fct(logits, labels)
        loss_t2i = loss_fct(logits.T, labels)
        loss3 = (loss_i2t + loss_t2i) / 2

        loss1 = category_softmax_loss(logits, batch["category"], batch["category"])
        loss2 = pairwise_ranking_loss(logits, batch["category"])
        loss = adaptive_loss(loss1, loss2, loss3)

        map_t2i = compute_map(logits, batch["category"], batch["category"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(matcher.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        best_map = max(best_map, map_t2i)

        if args.rank == 0 and batch["caption_input_ids"].shape[0] >= 32:
            print(f"[Step] Loss: {loss.item():.4f}, mAP: {map_t2i:.4f}")

    return total_loss / len(train_loader), best_map


def validate(encoder, matcher, val_loader, args):
    encoder.eval()
    matcher.eval()

    all_feats = {
        "text_embeds": [],
        "image_embeds": [],
        "text_seq": [],
        "image_patches": [],
        "img_des": [],
        "img_des_seq": [],
        "caption_cat": [],
        "img_cat": [],
        "caption_lab": [],
    }

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", disable=args.rank != 0):
            for key in batch:
                batch[key] = batch[key].to(args.device)

            outputs = encoder(
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                pixel_values=batch["pixel_values"],
                image_description_input_ids=batch["image_description_input_ids"],
                image_description_attention_mask=batch["image_description_attention_mask"],
            )

            all_feats["text_embeds"].append(outputs["text_embeds"])
            all_feats["image_embeds"].append(outputs["image_embeds"])
            all_feats["text_seq"].append(outputs["text_seq_tokens"])
            all_feats["image_patches"].append(outputs["image_patch_tokens"])
            all_feats["img_des"].append(outputs["image_description_embeds"])
            all_feats["img_des_seq"].append(outputs["image_description_seq_tokens"])
            all_feats["caption_cat"].append(batch["category"])
            all_feats["img_cat"].append(batch["category"])
            all_feats["caption_lab"].append(batch["label"])

    if args.rank == 0:
        for key in all_feats:
            all_feats[key] = torch.cat(all_feats[key], dim=0)

        logits = matcher(
            all_feats["img_des"],
            all_feats["img_des_seq"],
            all_feats["text_embeds"],
            all_feats["text_seq"],
            all_feats["image_embeds"],
            all_feats["image_patches"],
        )

        map_i2t = compute_map(logits.T, all_feats["img_cat"], all_feats["caption_cat"])
        map_t2i = compute_map(logits, all_feats["caption_cat"], all_feats["img_cat"])
        print(f"Validation mAP: Text->Image: {map_t2i:.4f}, Image->Text: {map_i2t:.4f}")
        return (map_t2i + map_i2t) / 2

    return 0.0


def main(args):
    args = init_distributed(args)

    encoder = DBMGEncoder(args).to(args.device)
    matcher = DBMG(text_dim=512, image_dim=args.hidden_dim, hidden_dim=args.hidden_dim).to(args.device)

    encoder = DDP(encoder, device_ids=None, output_device=args.rank, find_unused_parameters=True)
    matcher = DDP(matcher, device_ids=None, output_device=args.rank, find_unused_parameters=True)

    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(matcher.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )
    loss_fct = torch.nn.CrossEntropyLoss()
    adaptive_loss = AdaptiveLossWeighting().to(args.device)

    train_loader, val_loader, _ = build_dataloaders(args)

    best_val_map = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_map = train_one_epoch(
            epoch, encoder, matcher, train_loader, optimizer, loss_fct, adaptive_loss, args
        )
        val_map = validate(encoder, matcher, val_loader, args)

        if args.rank == 0:
            history.append((epoch, train_loss, train_map, val_map))
            print(
                f"[Epoch {epoch}] Loss: {train_loss:.4f}, "
                f"Train mAP: {train_map:.4f}, Val mAP: {val_map:.4f}"
            )
            if val_map > best_val_map:
                best_val_map = val_map
                no_improve = 0
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "encoder": encoder.state_dict(),
                        "matcher": matcher.state_dict(),
                    },
                    f"{args.checkpoint_dir}/best.pth",
                )
                print(f"Saved best model (val mAP: {best_val_map:.4f})")
            else:
                no_improve += 1
                if no_improve >= 8:
                    print("Early stopping")
                    break

        scheduler.step()

    if args.rank == 0:
        df = pd.DataFrame(history, columns=["Epoch", "Train Loss", "Train mAP", "Val mAP"])
        df.to_excel("training_log.xlsx", index=False)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="29500")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.9)
    parser.add_argument("--text_max_length", type=int, default=40)
    parser.add_argument("--hidden_dim", type=int, default=96)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    main(args)
