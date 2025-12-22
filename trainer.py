from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
def get_train_params():
    train_params = {
        'silent_mode': False,
        "average_best_models": False,
        "run_validation_freq": 5,
        "resume": True,
        "resume_path": r"C:\Users\User\PycharmProjects\IdeaFev\checkpoints\IdeaFev_experiment\RUN_20251221_144829_957193\ckpt_latest.pth",
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": False,
        # "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 40,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=6,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.6,
                top_k_predictions=150,
                num_cls=6,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.6,
                    nms_top_k=150,
                    max_predictions=50,
                    nms_threshold=0.5
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }
    return train_params