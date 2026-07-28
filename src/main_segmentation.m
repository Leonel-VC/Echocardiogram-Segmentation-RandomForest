function main_segmentation()
% MAIN_SEGMENTATION Main script for echocardiogram segmentation using Random Forest
%   This script loads pre-extracted features, trains a Random Forest classifier,
%   evaluates performance on test data, and visualizes results.
%
%   Workflow:
%       1. Load feature table and images
%       2. Prepare data for training (reshape, split)
%       3. Hyperparameter tuning with cross-validation
%       4. Train best model
%       5. Evaluate using medical image segmentation metrics
%       6. Visualize sample predictions

    clc;
    close all;
    
    % --- Configuration ---
    config.data_path = 'echonet-sample/frames/';
    config.mask_path = 'echonet-sample/masks/';
    config.attribute_file = 'AttributesTable.mat';
    config.num_train = 140;
    config.num_val_ratio = 0.15;
    config.random_seed = 42;
    config.image_size = [112, 112]; % Adjust based on your image size
    
    fprintf('=== Echocardiogram Segmentation using Random Forest ===\n\n');
    
    % --- 1. Load data ---
    fprintf('Loading data...\n');
    [images, data] = load_data(config);
    [n, m, num_images] = size(images);
    total_pixels = n * m;
    num_features = size(data, 3);
    
    fprintf('  Images loaded: %d\n', num_images);
    fprintf('  Image size: [%d, %d]\n', n, m);
    fprintf('  Features per pixel: %d\n', num_features);
    
    % --- 2. Prepare features and labels ---
    fprintf('\nPreparing features and labels...\n');
    X = reshape(data, [], num_features);
    
    % Load masks (assuming binary masks exist)
    Y = load_masks(config, num_images, n, m);
    
    % Create frame IDs
    frame_ids = repmat((1:num_images)', 1, total_pixels);
    frame_ids = reshape(frame_ids, [], 1);
    
    % --- 3. Split data ---
    fprintf('\nSplitting data into train/validation/test...\n');
    [X_train, Y_train, X_val, Y_val, X_test, Y_test, frame_ids_test] = ...
        split_data(X, Y, frame_ids, config);
    
    fprintf('  Training: %d pixels\n', size(X_train, 1));
    fprintf('  Validation: %d pixels\n', size(X_val, 1));
    fprintf('  Test: %d pixels\n', size(X_test, 1));
    
    % --- 4. Hyperparameter tuning ---
    fprintf('\n=== Hyperparameter Tuning ===\n');
    results_table = tune_hyperparameters(X_train, Y_train, X_val, Y_val);
    
    % Save results
    save('results_summary.mat', 'results_table');
    
    % --- 5. Train best model ---
    fprintf('\n=== Training Best Model ===\n');
    [best_model, best_idx] = select_best_model(results_table);
    fprintf('Best configuration: Trees=%d, Depth=%d, Criterion=%s\n', ...
        results_table(best_idx, 1), results_table(best_idx, 2), ...
        get_criterion_name(results_table(best_idx, 3)));
    
    rf_model = train_random_forest(X_train, Y_train, results_table(best_idx, :));
    
    % --- 6. Evaluate on test set ---
    fprintf('\n=== Evaluating on Test Set ===\n');
    Y_test_pred = predict(rf_model, X_test);
    Y_test_pred_binary = Y_test_pred == 1;
    Y_test_binary = Y_test == 1;
    
    metrics = compute_metrics(Y_test_pred_binary, Y_test_binary, ...
                              Y_test_pred, Y_test, frame_ids_test, n, m);
    
    % --- 7. Visualize results ---
    fprintf('\nGenerating visualizations...\n');
    visualize_predictions(images, Y_test, Y_test_pred, frame_ids_test, n, m);
    
    fprintf('\n=== DONE ===\n');
end

% -------------------------------------------------------------------------
function [images, data] = load_data(config)
% LOAD_DATA Loads images and feature table
    % Load images
    images = load_images(config.data_path);
    [n, m, ~] = size(images);
    
    % Load feature table
    if ~exist(config.attribute_file, 'file')
        error('Feature file not found: %s', config.attribute_file);
    end
    load(config.attribute_file, 'data');
    
    % Validate dimensions
    if size(data, 2) ~= n * m
        error('Feature table dimensions mismatch. Expected %d pixels, got %d.', ...
              n * m, size(data, 2));
    end
end

% -------------------------------------------------------------------------
function images = load_images(data_path)
% LOAD_IMAGES Loads all echocardiogram frames
    % Get dimensions from first image
    first_img = imread(fullfile(data_path, 'diastole0.png'));
    [n, m, ~] = size(first_img);
    
    total_images = 202;
    images = zeros(n, m, total_images, 'uint8');
    
    phase_names = ["diastole", "sistole"];
    for phase_idx = 1:length(phase_names)
        for i = 0:100
            img_idx = 101*(phase_idx-1) + i + 1;
            filename = sprintf('%s%d.png', phase_names(phase_idx), i);
            filepath = fullfile(data_path, filename);
            
            if exist(filepath, 'file')
                img = imread(filepath);
                images(:,:,img_idx) = rgb2gray(img);
            else
                warning('File not found: %s', filepath);
            end
        end
    end
end

% -------------------------------------------------------------------------
function Y = load_masks(config, num_images, n, m)
% LOAD_MASKS Loads binary segmentation masks
    Y = zeros(num_images, n*m);
    
    mask_path = config.mask_path;
    for i = 1:num_images
        % Assuming masks are named similarly to images
        % You may need to adjust this based on your actual mask naming
        phase_idx = ceil(i/101) - 1;
        frame_idx = mod(i-1, 101);
        phase_name = ["diastole", "sistole"];
        mask_file = sprintf('%s%d.png', phase_name(phase_idx+1), frame_idx);
        mask_path_full = fullfile(mask_path, mask_file);

        if exist(mask_path_full, 'file')
            mask = imread(mask_path_full);
            if size(mask, 3) == 3
                mask = rgb2gray(mask);
            end
            % Binarize if needed
            mask = mask > 128;
            Y(i,:) = mask(:);
        else
            warning('Mask not found: %s', mask_path_full);
        end
    end
end

% -------------------------------------------------------------------------
function [X_train, Y_train, X_val, Y_val, X_test, Y_test, frame_ids_test] = ...
    split_data(X, Y, frame_ids, config)
% SPLIT_DATA Splits data into train, validation, and test sets
    unique_frames = unique(frame_ids);
    num_frames = numel(unique_frames);
    
    % Reproducible random split
    rng(config.random_seed);
    idx = randperm(num_frames);
    
    num_train = config.num_train;
    num_val = round(config.num_val_ratio * num_frames);
    
    train_frames = unique_frames(idx(1:num_train));
    val_frames = unique_frames(idx(num_train+1:num_train+num_val));
    test_frames = unique_frames(idx(num_train+num_val+1:end));
    
    % Create logical indices
    train_idx = ismember(frame_ids, train_frames);
    val_idx = ismember(frame_ids, val_frames);
    test_idx = ismember(frame_ids, test_frames);
    
    % Shape coherence
    Y = reshape(Y, [], 1);

    % Extract data
    X_train = X(train_idx, :);
    Y_train = Y(train_idx, :);
    X_val = X(val_idx, :);
    Y_val = Y(val_idx, :);
    X_test = X(test_idx, :);
    Y_test = Y(test_idx, :);
    frame_ids_test = frame_ids(test_idx);
end

% -------------------------------------------------------------------------
function results = tune_hyperparameters(X_train, Y_train, X_val, Y_val)
% TUNE_HYPERPARAMETERS Tests multiple configurations and returns results
    % Hyperparameter combinations: [trees, max_depth, criterion]
    % criterion: 1 = 'gdi' (Gini), 0 = 'deviance' (Entropy)
    configs = [
        50, 500, 1;
        50, 50, 1;
        100, 500, 1;
        100, 50, 1;
        150, 500, 1;
        50, 500, 0;
        50, 50, 0;
        100, 500, 0;
        100, 50, 0;
        150, 500, 0
    ];

    num_configs = size(configs, 1);
    results = zeros(num_configs, 4); % [trees, depth, criterion, dice_score]
    results(:,1:3) = configs;
    
    fprintf('Testing %d configurations...\n', num_configs);
    
    for i = 1:num_configs
        fprintf('  Config %d/%d: Trees=%d, Depth=%d, Criterion=%s...', ...
            i, num_configs, configs(i,1), configs(i,2), ...
            get_criterion_name(configs(i,3)));
        
        try
            % Train model
            model = train_random_forest(X_train, Y_train, configs(i,:));
            
            % Predict on validation
            [Y_val_pred, ~ ] = predict(model, X_val);
            Y_val_pred_binary = Y_val_pred == 1;
            Y_val_binary = Y_val == 1;
            
            % Calculate Dice coefficient
            dice = 2 * sum((Y_val_pred_binary == 1) & (Y_val_binary == 1)) / ...
                   (sum(Y_val_pred_binary == 1) + sum(Y_val_binary == 1));
            results(i,4) = dice;
            
            fprintf(' Dice: %.4f\n', dice);
        catch ME
            fprintf(' ERROR: %s\n', ME.message);
            results(i,4) = 0;
        end
    end
end

% -------------------------------------------------------------------------
function model = train_random_forest(X_train, Y_train, config)
% TRAIN_RANDOM_FOREST Trains a Random Forest with given configuration
    trees = config(1);
    max_depth = config(2);
    criterion_idx = config(3);
    
    % Set criterion
    if criterion_idx == 1
        criterion = 'gdi'; % Gini impurity
    else
        criterion = 'deviance'; % Entropy
    end
    
    % Create template tree
    template = templateTree('MaxNumSplits', max_depth, ...
                            'SplitCriterion', criterion);
    
    % Train ensemble with bagging
    model = fitcensemble(X_train, Y_train, 'Method', 'Bag', ...
                        'NumLearningCycles', trees, ...
                        'Learners', template);
end

% -------------------------------------------------------------------------
function criterion_name = get_criterion_name(idx)
% GET_CRITERION_NAME Returns criterion name from index
    if idx == 1
        criterion_name = 'gdi';
    else
        criterion_name = 'deviance';
    end
end

% -------------------------------------------------------------------------
function [best_model, best_idx] = select_best_model(results)
% SELECT_BEST_MODEL Finds configuration with best Dice score
    [~, best_idx] = max(results(:,4));
    best_model = results(best_idx, :);
end

% -------------------------------------------------------------------------
function metrics = compute_metrics(Y_pred, Y_true, Y_pred_raw, Y_true_raw, ...
                                   frame_ids_test, n, m)
% COMPUTE_METRICS Calculates all segmentation metrics
    fprintf('\nComputing metrics...\n');
    
    % Convert to binary
    Y_pred_binary = Y_pred;
    Y_true_binary = Y_true;
    
    % Basic metrics
    TP = sum((Y_pred_binary == 1) & (Y_true_binary == 1));
    FP = sum((Y_pred_binary == 1) & (Y_true_binary == 0));
    FN = sum((Y_pred_binary == 0) & (Y_true_binary == 1));
    TN = sum((Y_pred_binary == 0) & (Y_true_binary == 0));
    
    % Dice Similarity Coefficient
    dice = 2 * TP / (2*TP + FP + FN);
    
    % Jaccard Index (IoU)
    jaccard = TP / (TP + FP + FN);
    
    % Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % Precision
    precision = TP / (TP + FP);
    
    % Recall (Sensitivity)
    recall = TP / (TP + FN);
    
    % Average Surface Distance (ASD)
    asd = compute_asd(Y_pred_binary, Y_true_binary, frame_ids_test, n, m);
    
    % Display results
    fprintf('  Dice Similarity Coefficient: %.4f\n', dice);
    fprintf('  Jaccard Index (IoU): %.4f\n', jaccard);
    fprintf('  Accuracy: %.4f\n', accuracy);
    fprintf('  Precision: %.4f\n', precision);
    fprintf('  Recall (Sensitivity): %.4f\n', recall);
    fprintf('  Average Surface Distance: %.4f pixels\n', asd);
    
    % Store in struct
    metrics = struct();
    metrics.dice = dice;
    metrics.jaccard = jaccard;
    metrics.accuracy = accuracy;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.asd = asd;
end

% -------------------------------------------------------------------------
function asd = compute_asd(Y_pred, Y_true, frame_ids_test, n, m)
% COMPUTE_ASD Calculates Average Surface Distance
    unique_frames = unique(frame_ids_test);
    num_frames = length(unique_frames);
    total_asd = 0;
    
    for k = 1:num_frames
        frame_idx = frame_ids_test == unique_frames(k);
        
        % Reshape to 2D
        gt = reshape(Y_true(frame_idx), [n, m]);
        pred = reshape(Y_pred(frame_idx), [n, m]);
        
        % Find boundaries
        gt_boundary = bwperim(gt);
        pred_boundary = bwperim(pred);
        
        % Get coordinates
        [gt_rows, gt_cols] = find(gt_boundary);
        [pred_rows, pred_cols] = find(pred_boundary);
        
        % Skip if no boundaries found
        if isempty(gt_rows) || isempty(pred_rows)
            continue;
        end
        
        % Compute pairwise distances
        distances = pdist2([gt_rows, gt_cols], [pred_rows, pred_cols]);
        
        % Minimum distances
        min_gt_to_pred = min(distances, [], 2);
        min_pred_to_gt = min(distances, [], 1);
        
        % Average distance for this frame
        frame_asd = (sum(min_gt_to_pred) + sum(min_pred_to_gt)) / ...
                    (length(gt_rows) + length(pred_rows));
        total_asd = total_asd + frame_asd;
    end
    
    asd = total_asd / num_frames;
end

% -------------------------------------------------------------------------
function visualize_predictions(images, Y_true, Y_pred, frame_ids_test, n, m)
% VISUALIZE_PREDICTIONS Displays sample segmentations
    unique_frames = unique(frame_ids_test);
    num_samples = min(20, length(unique_frames));
    sample_indices = randperm(length(unique_frames), num_samples);
    
    fprintf('Visualizing %d sample predictions...\n', num_samples);
    
    for k = 1:num_samples
        frame_idx = sample_indices(k);
        frame_num = unique_frames(frame_idx);
        pixel_idx = frame_ids_test == frame_num;
        
        % Reshape to 2D
        original = images(:,:,frame_num);
        gt = reshape(Y_true(pixel_idx), [n, m]);
        pred = reshape(Y_pred(pixel_idx), [n, m]);
        
        % Create figure
        figure('Name', sprintf('Frame %d', frame_num), 'Position', [100, 100, 1200, 400]);
        
        subplot(1, 3, 1);
        imshow(original, []);
        title('Original Frame', 'FontSize', 12);
        
        subplot(1, 3, 2);
        imshow(gt, []);
        title('Ground Truth Mask', 'FontSize', 12);
        
        subplot(1, 3, 3);
        imshow(pred, []);
        title('Predicted Segmentation', 'FontSize', 12);
        
        % Save figure
        saveas(gcf, sprintf('prediction_frame_%d.png', frame_num));
    end
end