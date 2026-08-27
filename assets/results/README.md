# Released experiment summaries

This directory contains only aggregate training curves and metric CSV files copied from the original experiment outputs. Raw datasets, per-image label/prediction previews, and checkpoints are deliberately excluded.

```text
results/
├── ribfrac/
│   ├── wdr_moe/
│   ├── yolo_master_moe_loss/
│   ├── yolov11/
│   ├── yolov12/
│   ├── yolov5_csff/
│   └── wcay/
└── fracatlas/
    ├── wdr_moe/
    ├── yolo_master/
    ├── yolov11/
    ├── yolov12/
    ├── yolov5_csff/
    └── wcay/
```

The manuscript tables contain the official rounded comparison values. CSV files are retained as experiment provenance and may include values from every epoch; do not replace paper-table results with a cross-epoch maximum without documenting that evaluation choice.
