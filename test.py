from super_gradients.training import Trainer, models
from trainer import get_train_params
from dataloader import get_dataset_params
from dataloader import dataloader
from super_gradients import setup_device
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import random
import sys
def test(test_data):
    best_model = models.get(
        'yolo_nas_s',
        num_classes=6,
        checkpoint_path="./checkpoints/IdeaFev_experiment/RUN_20251218_135014_616681/ckpt_latest.pth"
    )

    checkpoints_dir = './checkpoints'
    trainer = Trainer(experiment_name="IdeaFev_test_only", ckpt_root_dir=checkpoints_dir)

    print(1)
    print("len(test_data):", len(test_data))
    metrics = DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=6,
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
    )
    result = trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=[metrics]
    )
    metrics_result = metrics.compute()

    print("TEST METRICS:", metrics_result, file=sys.stderr)

    with open("test_metrics.json", "w") as f:
        f.write(str(metrics_result))


if __name__ == '__main__':
    dataset_params = get_dataset_params()
    train_data, test_data = dataloader(dataset_params)
    test(test_data)
