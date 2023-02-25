# The new config inherits a base config to highlight the necessary modification
_base_ = './point_rend/point-rend_r50-caffe_fpn_ms-3x_coco.py'


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
            point_head=dict(num_classes=1),
            mask_head=dict(num_classes=1)
        )
    )

# Modify dataset related settings
data_root = '/content/mmdetection/surfaces/'
metainfo = {
    'classes': ('surface', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_3x_coco/point_rend_r50_caffe_fpn_mstrain_3x_coco-e0ebb6b7.pth'

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=12,
    val_interval=12,
    )

optim_wrapper = dict(
        optimizer = dict(lr=0.02 / 8)
    )