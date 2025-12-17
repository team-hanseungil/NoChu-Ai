import glob
from super_gradients.training.dataloaders.dataloaders import (coco_detection_yolo_format_train, coco_detection_yolo_format_val)
import os
def get_dataset_params():
    dataset_params={
        'data_dir': "C:/face_data",
        'train_image_dir': "C:/face_data/Training/images/",
        'train_label_dir': "C:/face_data/Training/labels/",
        'val_image_dir': "C:/face_data/Validation/images/",
        'val_label_dir': "C:/face_data/Validation/labels/",
        'test_image_dir': "C:/face_data/Validation/images/",
        'test_label_dir': "C:/face_data/Validation/labels/",
        'classes':['기쁨', '당황', '분노', '불안', '상처', '슬픔'],
        "input_dim": [360, 360],
        'batch_size': 32
    }
    return dataset_params

def dataloader(params):
    train_data=coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': params['data_dir'],
            'images_dir': params['train_image_dir'],
            'labels_dir': params['train_label_dir'],
            'classes': params['classes'],
        },
        dataloader_params={
            'batch_size': params['batch_size'],
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True
        }
    )
    val_data=coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': params['data_dir'],
            'images_dir': params['val_image_dir'],
            'labels_dir': params['val_label_dir'],
            'classes': params['classes'],
        },
        dataloader_params={
            'batch_size': params['batch_size'],
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': True
        }
    )
    # test_data=coco_detection_yolo_format_val(
    #     dataset_params={
    #         'data_dir': params['data_dir'],
    #         'images_dir': params['test_image_dir'],
    #         'labels_dir': params['test_label_dir'],
    #         'classes': params['classes'],
    #     },
    #     dataloader_params={
    #         'batch_size': params['batch_size'],
    #         'num_workers': 4
    #     }
    # )
    return train_data, val_data