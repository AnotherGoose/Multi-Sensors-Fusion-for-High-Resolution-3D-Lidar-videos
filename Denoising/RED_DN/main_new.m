clc;
clear;
close all;

% configure the path
% denoising functions
% addpath(genpath('./tnrd_denoising/'));
% SD, FP, and ADMM methods
addpath(genpath('./minimizers/'));
% contains the default params
addpath(genpath('./parameters/'));
% contains basic functions
addpath(genpath('./helper_functions/'));
% test images for the debluring and super resolution problems, 
% taken from NCSR software package
addpath(genpath('./test_images/'));

% set light_mode = true to run the code in a sub optimal but faster mode
% set light_mode = false to obtain the results reported in the RED paper
light_mode = false;

if light_mode
    fprintf('Running in light mode. ');
    fprintf('Turn off to obatain the results reported in RED paper.\n');
else
    fprintf('Light mode option is off. ');
    fprintf('Reproducing the result in RED paper.\n');
end

%% read the original image

v = VideoReader('Frames_15fps.mp4');
frame = read(v,50);
orig_im = double(rgb2gray(frame));


%{
file_name = 'barbara.tif';
orig_im = imread(['RED/test_images/' file_name]);
orig_im = double(orig_im);
%}
%% define the degradation model

% choose the secenrio: 'UniformBlur', 'GaussianBlur', or 'Downscale'
degradation_model = 'GaussianBlur';

fprintf('Test case: %s degradation model.\n', degradation_model);

switch degradation_model
    case 'UniformBlur'
        % noise level
        input_sigma = sqrt(2);
        % filter size
        psf_sz = 9;
        % create uniform filter
        psf = fspecial('average', psf_sz);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function-handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'GaussianBlur'
        % noise level
        input_sigma = 4;
        % filter size
        psf_sz = 25;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % use fft to solve a system of linear equations in closed form
        use_fft = true;
        % create a function handle to blur the image
        ForwardFunc = ...
            @(in_im) imfilter(in_im,psf,'conv','same','circular');
        % the psf is symmetric, i.e., the ForwardFunc and BackwardFunc
        % are the same
        BackwardFunc = ForwardFunc;
        % special initialization (e.g. the output of other method)
        % set to identity mapping
        InitEstFunc = @(in_im) in_im;
        
    case 'Downscale'
        % noise level
        input_sigma = sqrt(2);
        % filter size
        psf_sz = 7;
        % std of the Gaussian filter
        gaussian_std = 1.6;
        % create gaussian filter
        psf = fspecial('gaussian', psf_sz, gaussian_std);
        % scaling factor
        scale = 3;
        
        % compute the size of the low-res image
        lr_im_sz = [ceil(size(orig_im,1)/scale),...
                    ceil(size(orig_im,2)/scale)];        
        % create the degradation operator
        H = CreateBlurAndDecimationOperator(scale,lr_im_sz,psf);
        % downscale
        ForwardFunc = @(in_im) reshape(H*in_im(:),lr_im_sz);        
        % upscale
        BackwardFunc = @(in_im) reshape(H'*in_im(:),scale*lr_im_sz);
        % special initialization (e.g. the output of other method)
        % use bicubic upscaler
        InitEstFunc = @(in_im) imresize(in_im,scale,'bicubic');
    
        
    otherwise
        error('Degradation model is not defined');
end


%% degrade the original image

switch degradation_model
    case {'UniformBlur', 'GaussianBlur'}
        fprintf('Blurring...');
        % blur each channel using the ForwardFunc
        input_im = zeros( size(orig_im) );
        for ch_id = 1:size(orig_im,3)
            input_im(:,:,ch_id) = ForwardFunc(orig_im(:,:,ch_id));
        end
        % use 'seed' = 0 to be consistent with the experiments in NCSR
        randn('seed', 0);

    case 'Downscale'
        fprintf('Downscaling...');
        % blur the image, similar to the degradation process of NCSR
        input_im = Blur(orig_im, psf);
        % decimate
        input_im = input_im(1:scale:end,1:scale:end,:);
        % use 'state' = 0 to be consistent with the experiments in NCSR
        randn('state', 0);

    otherwise
        error('Degradation model is not defined');
end

% add noise
fprintf(' Adding noise...');
input_im = input_im + input_sigma*randn(size(input_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);

if strcmp(degradation_model,'Downscale')
    % upscale using bicubic
    input_im = imresize(input_im,scale,'bicubic');
    input_im = input_im(1:size(orig_im,1), 1:size(orig_im,2), :); 
end
fprintf(' Done.\n');
psnr_input = ComputePSNR(orig_im, input_im);


%% minimize the Laplacian regularization functional

% choose the regularization functional: 'ADMM','Fixed-Point','Steepest-Descent
reg_function = 'ADMM';

% choose the denoising engine to be used: 'tnrd','mat_DnCNN','Med3','Wnr'
denoising_eng = 'Med3';

switch reg_function
    case 'Fixed-Point'
        
        fprintf('Restoring using RED: Fixed-Point method\n');

        switch degradation_model
            case 'UniformBlur'
                params_fp = GetUniformDeblurFPParams(light_mode, psf, use_fft);
            case 'GaussianBlur'
                params_fp = GetGaussianDeblurFPParams(light_mode, psf, use_fft);
            case 'Downscale'
                assert(exist('use_fft','var') == 0);
                params_fp = GetSuperResFPParams(light_mode);
            otherwise
                error('Degradation model is not defined');
        end
        tStart = cputime;
        [est_fp_im, psnr_fp] = RunFP(input_luma_im,...
                                     ForwardFunc,...
                                     BackwardFunc,...
                                     InitEstFunc,...
                                     input_sigma,...
                                     params_fp,...
                                     orig_luma_im,...
                                     denoising_eng);
        out_fp_im = MergeChannels(input_im,est_fp_im);
        output_im = out_fp_im;
        psnr_out = psnr_fp;
        fprintf('Done.\n');

    case 'ADMM'
        
        fprintf('Restoring using RED: ADMM method\n');
        tStart = cputime;
        switch degradation_model
            case 'UniformBlur'
                params_admm = GetUniformDeblurADMMParams(light_mode, psf, use_fft);
            case 'GaussianBlur'
                params_admm = GetGaussianDeblurADMMParams(light_mode, psf, use_fft);
            case 'Downscale'
                assert(exist('use_fft','var') == 0);
                params_admm = GetSuperResADMMParams(light_mode);        
            otherwise
                error('Degradation model is not defined');
        end

        [est_admm_im, psnr_admm] = RunADMM(input_luma_im,...
                                           ForwardFunc,...
                                           BackwardFunc,...
                                           InitEstFunc,...
                                           input_sigma,...
                                           params_admm,...
                                           orig_luma_im,...
                                           denoising_eng);
        out_admm_im = MergeChannels(input_im,est_admm_im);
        output_im = out_admm_im;
        psnr_out = psnr_admm;
        fprintf('Done.\n');
        
    case 'Steepest-Descent'        
        fprintf('Restoring using RED: Steepest-Descent method\n');
        tStart = cputime;
        switch degradation_model
            case 'UniformBlur'
                params_sd = GetUniformDeblurSDParams(light_mode);
            case 'GaussianBlur'
                params_sd = GetGaussianDeblurSDParams(light_mode);
            case 'Downscale'
                params_sd = GetSuperResSDParams(light_mode);
            otherwise
                error('Degradation model is not defined');
        end

        [est_sd_im, psnr_sd] = RunSD(input_luma_im,...
                                     ForwardFunc,...
                                     BackwardFunc,...
                                     InitEstFunc,...
                                     input_sigma,...
                                     params_sd,...
                                     orig_luma_im,...
                                     denoising_eng);
        % convert back to rgb if needed
        out_sd_im = MergeChannels(input_im,est_sd_im);
        output_im = out_sd_im;
        psnr_out = psnr_sd;
        fprintf('Done.\n');
    otherwise 
        error('Regularization function not defined')
end
orig_im = double(orig_im);
tEnd = cputime - tStart;
ssim_calc = ssim(orig_im, output_im);
abs_difr_im = imabsdiff(orig_im, output_im);
abs_difr_mean = mean(abs_difr_im,'all');
 
RMSE_Val =  sqrt(mean(mean((output_im - orig_im).^2)));

%% Display images

fprintf('Absolute Difference = %f \n', abs_difr_mean);
fprintf('Noise Level = %f \n', input_sigma);
fprintf('Input PSNR = %f \n', psnr_input);
fprintf('RED: ADMM PSNR = %f \n', psnr_out);
fprintf('Computation time = %f \n', tEnd);
fprintf('Structural Similarity = %f \n', ssim_calc);

save('datatemplate', 'abs_difr_mean',...
                    'input_sigma',...
                    'psnr_input',...
                    'psnr_input',...
                    'psnr_out',...
                    'tEnd',...
                    'ssim_calc'); 
                


