from glob import glob
from pprint import pprint

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import (
    Compose,
    ElasticTransform,
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    Rotate,
    VerticalFlip,
)
from albumentations.augmentations import transforms as T
from albumentations.augmentations.blur import transforms as blur
from albumentations.augmentations.dropout.channel_dropout import ChannelDropout
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout
from albumentations.augmentations.dropout.grid_dropout import GridDropout
from albumentations.augmentations.dropout.mask_dropout import MaskDropout
from albumentations.augmentations.geometric.rotate import RandomRotate90
from datasets import Dataset, Image
from transformers import (
    EarlyStoppingCallback,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
)


def train_transforms(example_batch):
    images = []
    labels = []
    for i in range(len(example_batch["pixel_values"])):
        img = example_batch["pixel_values"][i]
        mask = example_batch["label"][i]
        img = img.resize((256, 256))  # Resize the image
        mask = mask.resize((256, 256))  # Resize the mask
        augmented = train_augs(image=np.asarray(img), mask=np.asarray(mask))
        images.append(augmented["image"])
        labels.append(augmented["mask"])
    inputs = train_processor(images, labels, return_tensors="pt")
    return inputs


def val_transforms(example_batch):
    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = test_processor(images, labels, return_tensors="pt")
    return inputs


metric = evaluate.load("mean_iou")


@torch.no_grad()
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = F.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric._compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=-100,
    )
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update(
        {f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
    )
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics


def evaluate_experiment(exp_name, datasets):
    model = SegformerForSemanticSegmentation.from_pretrained(
        exp_name, id2label=id2label, label2id=label2id
    ).eval()  # type: ignore
    evaluator = Trainer(model=model, compute_metrics=compute_metrics)  # type: ignore

    round_metrics = lambda d: {k: round(v, 4) for k, v in d.items()}
    print("========================================================")

    print("========= Evaluating on train set... =========")
    metrics = evaluator.evaluate(datasets[0])  # type: ignore
    metrics = round_metrics(metrics)
    pprint(metrics)

    print("========= Evaluating on validation set... =========")
    metrics = evaluator.evaluate(datasets[1])  # type: ignore
    metrics = round_metrics(metrics)
    pprint(metrics)

    print("========= Evaluating on test set... =========")
    metrics = evaluator.evaluate(datasets[2])  # type: ignore
    metrics = round_metrics(metrics)
    pprint(metrics)


def b2_model_init():
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        id2label=id2label,
        label2id=label2id,
        classifier_dropout_prob=0.3,
    )


def b4_model_init():
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b4",
        id2label=id2label,
        label2id=label2id,
        classifier_dropout_prob=0.3,
    )


if __name__ == "__main__":
    IMAGE_DIR = "data/images"
    image_files = sorted(glob(f"{IMAGE_DIR}/*.jpg"))
    image_files = sorted(list(filter(lambda x: not x.endswith("(1).jpg"), image_files)))

    MASK_DIR = "data/masks"
    mask_files = sorted(glob(f"{MASK_DIR}/*.tiff"))
    mask_files = sorted(list(filter(lambda x: not x.endswith("(1).tiff"), mask_files)))

    original_ds = Dataset.from_dict({"pixel_values": image_files, "label": mask_files})
    original_ds = original_ds.cast_column("pixel_values", Image())
    original_ds = original_ds.cast_column("label", Image())

    original_ds = Dataset.from_dict({"pixel_values": image_files, "label": mask_files})
    original_ds = original_ds.cast_column("pixel_values", Image())
    original_ds = original_ds.cast_column("label", Image())

    original_ds = original_ds.shuffle(seed=1)
    ds = original_ds.train_test_split(test_size=0.3)
    train_ds = ds["train"]
    evaluation_ds = ds["test"].train_test_split(test_size=0.5)
    valid_ds = evaluation_ds["train"]
    test_ds = evaluation_ds["test"]

    id2label = {0: "background", 1: "sandstone", 2: "mudstone"}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    train_processor = SegformerImageProcessor(do_resize=False)
    test_processor = SegformerImageProcessor(size={"width": 256, "height": 256})

    # Define the Albumentations transformations
    p_max = 0.4
    p_min = 0.2
    train_augs = Compose(
        [
            HorizontalFlip(p=p_max),
            VerticalFlip(p=p_max),
            RandomRotate90(p=p_min),
            Rotate(limit=30, p=p_min),
            # ElasticTransform(
            #     p=p_min, alpha=60, sigma=60 * 0.05, alpha_affine=60 * 0.03
            # ),
            RandomBrightnessContrast(p=p_max),
            # RandomGamma(p=p_min),
            # T.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=p_max),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # blur.AdvancedBlur(p=p_max),
            T.GaussNoise(var_limit=(10.0, 30.0), mean=0, per_channel=True, p=p_max),
            T.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=p_max),
            # T.MultiplicativeNoise(
            #     multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=p_max
            # ),
            # T.FancyPCA(p=p_max),
            # T.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=p_max),
            # T.Downscale(scale_min=0.7, scale_max=0.9, interpolation=2, p=p_max),
            # T.RandomShadow(
            #     num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=3, p=p_min
            # ),
            # T.RandomToneCurve(p=p_max),
            # MaskDropout(
            #     max_objects=1,
            #     image_fill_value=0,
            #     mask_fill_value=0,
            #     always_apply=False,
            #     p=p_min,
            # ),
            CoarseDropout(p=p_min),
            # ChannelDropout(p=p_min),
            GridDropout(p=p_min),
        ]
    )

    # Set transforms
    original_ds.set_transform(val_transforms)
    train_ds.set_transform(train_transforms)
    valid_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    ## hyperparams
    lr = 1e-4
    batch_size = 32
    wd = 5e-3
    variant = "b4"
    warmup = 5e-2
    scheduler = "constant"
    # exp_name = f"prod-{variant}-lr={lr}-augnoise_{int(p_min * 10)}_{int(p_max * 10)}-wd={wd}-warmup={warmup}-sched={scheduler}"
    exp_name = (
        f"prod-{variant}-lr={lr}-lessaug-wd={wd}-warmup={warmup}-sched={scheduler}"
    )

    training_args = TrainingArguments(
        # fixed args
        f"experiments/{exp_name}",
        logging_dir=f"logs/{exp_name}",
        metric_for_best_model="mean_iou",
        load_best_model_at_end=True,
        save_total_limit=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        eval_steps=10,
        logging_steps=10,
        num_train_epochs=100,
        per_device_eval_batch_size=8,
        # hyperparams
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        warmup_ratio=warmup,
        weight_decay=wd,
        lr_scheduler_type=scheduler,
    )

    trainer = Trainer(
        model_init=b2_model_init if variant == "b2" else b4_model_init,  # type: ignore
        args=training_args,
        train_dataset=original_ds,  # type: ignore
        eval_dataset=original_ds,  # type: ignore
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=20, early_stopping_threshold=0.001
            )
        ],
    )

    trainer.train()
    trainer.save_model()
    evaluate_experiment(f"experiments/{exp_name}", [train_ds, valid_ds, test_ds])
