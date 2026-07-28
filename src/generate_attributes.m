function generate_attributes(data_path, output_file)
% GENERATE_ATTRIBUTES Extracts per-pixel features from echocardiogram frames
%   generate_attributes(data_path, output_file) processes all images in 
%   the specified path and saves a feature matrix for Random Forest training.
%
%   Inputs:
%       data_path   - String, path to folder containing images (default: 'echonet-sample/frames/')
%       output_file - String, name of output .mat file (default: 'AttributesTable.mat')
%
%   Features extracted per pixel:
%       1. Intensity
%       2. X-position (row)
%       3. Y-position (column)
%       4. Local mean (3x3 window)
%       5. Local median (3x3 window)
%       6. Local max (3x3 window)
%       7. Local min (3x3 window)
%       8. Local variance (3x3 window)
%       9. Anisotropic diffusion filter response
%
%   Output:
%       Saves 'data' matrix of size [numImages, n*m, 9] to output_file

    % Default arguments
    if nargin < 1
        data_path = 'echonet-sample/frames/';
    end
    if nargin < 2
        output_file = 'AttributesTable.mat';
    end

    % Validate path
    if ~exist(data_path, 'dir')
        error('Directory not found: %s', data_path);
    end

    fprintf('Loading images from: %s\n', data_path);
    
    % Get first image to determine dimensions
    first_img_path = fullfile(data_path, 'diastole0.png');
    if ~exist(first_img_path, 'file')
        error('First image not found: %s', first_img_path);
    end
    
    first_image = imread(first_img_path);
    [n, m, ~] = size(first_image);
    
    % Preallocate image stack (202 images: 101 diastole + 101 systole)
    total_images = 202;
    images = zeros(n, m, total_images, 'uint8');
    
    % Load images
    image_names = ["diastole", "sistole"]; % Corrected spelling
    for phase_idx = 1:length(image_names)
        phase_name = image_names(phase_idx);
        for i = 0:100
            img_idx = 101*(phase_idx-1) + i + 1;
            filename = sprintf('%s%d.png', phase_name, i);
            filepath = fullfile(data_path, filename);
            
            if ~exist(filepath, 'file')
                warning('File not found: %s. Skipping...', filepath);
                continue;
            end
            
            img = imread(filepath);
            images(:,:,img_idx) = rgb2gray(img);
        end
    end
    
    num_images = size(images, 3);
    num_features = 9;
    
    % Preallocate feature matrix
    data = zeros(num_images, n*m, num_features);
    
    fprintf('Extracting features from %d images...\n', num_images);
    
    % Process each image
    for img_idx = 1:num_images
        if mod(img_idx, 20) == 0
            fprintf('Processing image %d/%d...\n', img_idx, num_images);
        end
        
        % Convert to double for calculations
        mask = double(images(:,:,img_idx));
        
        % --- Basic features ---
        intensity = mask(:);
        xpos = repmat((1:n)', 1, m);
        ypos = repmat((1:m), n, 1);
        
        % --- Local statistical features (3x3 window) ---
        mean_img = colfilt(mask, [3 3], 'sliding', @mean);
        median_img = colfilt(mask, [3 3], 'sliding', @median);
        max_img = colfilt(mask, [3 3], 'sliding', @max);
        min_img = colfilt(mask, [3 3], 'sliding', @min);
        var_img = colfilt(mask, [3 3], 'sliding', @var);
        
        % --- Anisotropic diffusion feature ---
        aniso_img = anisotropic_filter(mask);
        
        % Store all features
        data(img_idx, :, 1) = intensity;
        data(img_idx, :, 2) = xpos(:);
        data(img_idx, :, 3) = ypos(:);
        data(img_idx, :, 4) = mean_img(:);
        data(img_idx, :, 5) = median_img(:);
        data(img_idx, :, 6) = max_img(:);
        data(img_idx, :, 7) = min_img(:);
        data(img_idx, :, 8) = var_img(:);
        data(img_idx, :, 9) = aniso_img(:);
    end
    
    % Save the data
    save(output_file, 'data');
    fprintf('Features saved to: %s\n', output_file);
    fprintf('Matrix size: [%d, %d, %d]\n', size(data));
end

% -------------------------------------------------------------------------
function aniso_img = anisotropic_filter(mask)
% ANISOTROPIC_FILTER Applies anisotropic diffusion filtering
%   Applies a spatially-varying Gaussian filter based on local gradient
%   to preserve edges while smoothing homogeneous regions.
%
%   Input:
%       mask - 2D double matrix (image)
%   Output:
%       aniso_img - Filtered image of same size

    [n, m] = size(mask);
    
    % Laplacian kernels
    xKer = [0 0 0; 1 -2 1; 0 0 0];
    yKer = [0 1 0; 0 -2 0; 0 1 0];
    
    % Convolve with Laplacian kernels
    mLapX = conv2(mask, xKer, 'same');
    mLapY = conv2(mask, yKer, 'same');
    
    % Calculate orientation and scaling
    ca = mLapX + mLapY;
    ca(ca == 0) = eps; % Avoid division by zero
    
    a = ((mLapX - mLapY) ./ ca).^2;
    
    % Frobenius norm for contrast normalization
    nIm2 = norm(mask, 'fro');
    C = (1 - a) * nIm2;
    
    % Compute local variance
    var_img = colfilt(mask, [3 3], 'sliding', @var);
    var_img = reshape(var_img, n, m);
    
    % Calculate sigmas for anisotropic filter
    sig1 = (var_img .* (1 - a)) ./ (1 + C) * 50;
    sig1(sig1 == 0) = eps;
    
    sig2 = var_img ./ (1 + C) * 50;
    sig2(sig2 == 0) = eps;
    
    % Apply anisotropic filtering
    aniso_img = zeros(n, m);
    
    % Pad mask for boundary handling
    padded_mask = padarray(mask, [1 1], 'replicate');
    
    for i = 2:n-1
        for j = 2:m-1
            % Extract 3x3 window
            window = padded_mask(i-1:i+1, j-1:j+1);
            
            % Create Gaussian kernel with anisotropic sigmas
            [y_grid, x_grid] = ndgrid(-1:1, -1:1);
            h = exp(-((x_grid.^2 + y_grid.^2) ./ (2 * sig1(i,j)^2) + ...
                     (x_grid.^2 + y_grid.^2) ./ (2 * sig2(i,j)^2)));
            
            % Threshold small values for numerical stability
            h(h < eps * max(h(:))) = 0;
            
            % Normalize
            sumh = sum(h(:));
            if sumh ~= 0
                h = h / sumh;
            end
            
            % Apply filter
            aniso_img(i,j) = sum(window(:) .* h(:));
        end
    end
end