from super_gradients.training import Trainer, models
from trainer import get_train_parmas
from dataloader import get_dataset_params
from dataloader import dataloader
from super_gradients import setup_device
def main():
    model_arch = 'yolo_nas_s'
    max_epochs = 50
    checkpoints_dir = './checkpoints'
    experiment_name = './IdeaFev_experiment'
    setup_device(device="cuda")
    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=checkpoints_dir)

    dataset_params = get_dataset_params()
    train_data, val_data = dataloader(dataset_params)

    model = models.get(
        model_arch,
        num_classes=6,
        pretrained_weights="coco"
    )

    trainer.train(
        model=model,
        training_params=get_train_parmas(),
        train_loader=train_data,
        valid_loader=val_data
    )
if __name__ == '__main__':
    main()