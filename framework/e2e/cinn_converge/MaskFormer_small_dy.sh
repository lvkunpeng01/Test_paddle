python main.py -c paddlex/configs/modules/semantic_segmentation/MaskFormer_small.yaml \
    -o Global.mode=train -o Global.dataset_dir=../ADEChallengeData2016 \
    -o Train.num_classes=150 -o Train.epochs_iters=160000 -o Train.batch_size=4 \
    -o Train.warmup_steps=1500 -o Train.learning_rate=0.00006 \
    -o Global.device=gpu:0,1,2,3,4,5,6,7 -o Train.pretrain_weight_path=None -o Global.output='./output/semantic_segmentation/MaskFormer_small_dy'