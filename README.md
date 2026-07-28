# Echocardiogram-Segmentation-RandomForest

Machine learning-based segmentation of echocardiograms using Random Forest with pixel-wise feature extraction. This project segments the left ventricle in echocardiogram frames using a Random Forest classifier trained on per-pixel features including intensity, position, local statistics, and anisotropic filtering.

## Results Summary

| Metric | Value |
|--------|-------|
| **Best Configuration** | Trees=100, Depth=500, Criterion=Gini |
| **Validation Dice Score** | 0.7120 |
| **Test Dice Score** | 0.6752 |
| **Test IoU (Jaccard Index)** | 0.5096 |
| **Test Accuracy** | 95.08% |
| **Test Precision** | 82.27% |
| **Test Recall (Sensitivity)** | 57.25% |
| **Average Surface Distance (ASD)** | 4.00 pixels |

### Performance Interpretation

- **Dice Score (0.6752)**: Moderate segmentation performance, clinically useful for initial screening and analysis
- **High Precision (82.27%)**: Very few false positives, indicating the model is conservative and avoids over-segmentation
- **Moderate Recall (57.25%)**: Some under-segmentation occurs, suggesting areas for improvement
- **ASD (4.00 pixels)**: Average boundary error of ~4 pixels, reasonable for echocardiogram analysis

## Key Features

- **Pixel-wise feature extraction** including:
  - Intensity values
  - Spatial coordinates (x, y positions)
  - Local statistics (mean, median, max, min, variance in 3x3 window)
  - Anisotropic diffusion filter response
- **Random Forest classification** with hyperparameter optimization
- **Comprehensive evaluation** using medical imaging metrics:
  - Dice Similarity Coefficient
  - Jaccard Index (IoU)
  - Accuracy, Precision, Recall, Specificity
  - Average Surface Distance (ASD)
- **Automatic visualization** of predictions

## Getting Started

### Prerequisites

- **MATLAB R2020a** or later
- Required toolboxes:
  - Image Processing Toolbox
  - Statistics and Machine Learning Toolbox

### Running the Pipeline

**Step 1: Extract Features**
```matlab
% Run this once to generate the feature table
generate_attributes('data/frames/', 'AttributesTable.mat');
```

**Step 2: Run Segmentation**
```matlab
% Run the main segmentation pipeline
main_segmentation();
```

## Configuration

The main configuration parameters in ```main_segmentation.m```:
```matlab
config.data_path = 'data/frames/';             % Path to echocardiogram frames
config.mask_path = 'data/masks/';              % Path to ground truth masks
config.attribute_file = 'AttributesTable.mat'; % Feature file name
config.num_train = 140;                        % Number of training frames
config.num_val_ratio = 0.15;                   % Validation set ratio
config.random_seed = 42;                       % For reproducibility
config.image_size = [112, 112];                % Image dimensions
```

## Hyperparameter Tuning Results

| Config | Trees | Max Depth | Criterion |	Validation Dice |
|------|------|------|------|------|
| 1 | 50 | 500 | Gini | 0.6969 |
| 2 | 50 | 50 | Gini | 0.5082 |
| 3 | 100 | 500 | Gini | 0.7120 |
| 4 | 100 | 50 | Gini | 0.5263 |
| 5 | 150 | 500 | Gini | 0.6842 |
| 6 | 50 | 500 | Deviance | 0.6719 |
| 7 | 50 | 50 | Deviance | 0.5462 |
| 8 | 100 | 500 | Deviance | 0.6922 |
| 9 | 100 | 50 | Deviance | 0.5033 |
| 10 | 150 | 500 | Deviance | 0.7110 |

Best Configuration: Trees=100, Depth=500, Criterion=Gini

## Repository Structure

```
├── data/
│ ├── frames/
│ │ ├── diastole0.png ... diastole100.png
│ │ └── sistole0.png ... sistole100.png
| └── masks/
│ │ ├── diastole0.png ... diastole100.png
│ │ └── sistole0.png ... sistole100.png
├── results/                     # Results generated automatically
│ ├── results_summary.mat
│ ├── prediction_frame_54.png
│ ├── prediction_frame_56.png
│ ├── prediction_frame_155.png
│ ├── prediction_frame_192.png
│ └── prediction_frame_193.png
├── src/
│ ├── generate_attributes.m      # Feature extraction (Code 1)
│ └── main_segmentation.m        # Main pipeline (Code 2)
└── README.md                    # This file
```

## Output Files
After running the pipeline, you'll find:

| File | Description |
|------|------|
| ``` results_summary.mat ``` | All tested configurations and their Dice scores |
| ``` prediction_frame_*.png ``` | Individual frame visualizations |

## Methodology

### Feature Extraction

For each pixel in every image, we extract 9 features:

1. **Intensity** - Original pixel value
2. **X-Position** - Row coordinate
3. **Y-Position** - Column coordinate
4. **Local Mean** - Mean in 3x3 neighborhood
5. **Local Median** - Median in 3x3 neighborhood
6. **Local Max** - Maximum in 3x3 neighborhood
7. **Local Min** - Minimum in 3x3 neighborhood
8. **Local Variance** - Variance in 3x3 neighborhood
9. **Anisotropic Filter** - Edge-preserving smoothing response

### Model Training

- **Algorithm**: Random Forest with Bagging
- **Training Data**: 140 frames (70% of dataset)
- **Validation**: 15% for hyperparameter tuning
- **Test**: Remaining 15% for final evaluation

### Evaluation Metrics

- **Dice Coefficient**: Measures overlap between predicted and ground truth
- **Jaccard Index (IoU)**: Intersection over union
- **Accuracy**: Overall correct predictions
- **Precision/Recall**: Balance of false positives vs false negatives
- **Average Surface Distance**: Boundary alignment quality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The dataset is provided for academic research purposes only; please cite the original authors.

## Acknowledgments

* Echonet dataset for providing echocardiogram data
