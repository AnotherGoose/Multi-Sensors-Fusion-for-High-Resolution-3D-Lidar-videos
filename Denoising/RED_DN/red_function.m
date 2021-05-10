%% RED as function Script 
function [output_im,deg_im,tEnd,psnr_out,ssim_out,abs_diff,psnr_input,RMSE_val,RMSE_noisy_val] = red_function(deg_model,noise_level,orig_im,reg_function,denoising_eng,light_mode)
%% define the degradation model

switch deg_model
    case 'UniformBlur'
        % noise level
        input_sigma = noise_level;
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
        input_sigma = noise_level;
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
        input_sigma = noise_level;
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
    
    case 'no_degradation'
        
        
    otherwise
        error('Degradation model is not defined');
end


%% degrade the original image

switch deg_model
    case {'UniformBlur', 'GaussianBlur'}
        % fprintf('Blurring...');
        % blur each channel using the ForwardFunc
        input_im = zeros( size(orig_im) );
        for ch_id = 1:size(orig_im,3)
            input_im(:,:,ch_id) = ForwardFunc(orig_im(:,:,ch_id));
        end
        % use 'seed' = 0 to be consistent with the experiments in NCSR
        randn('seed', 0);

    case 'Downscale'
        % fprintf('Downscaling...');
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
% fprintf(' Adding noise...');
input_im = input_im + input_sigma*randn(size(input_im));

% convert to YCbCr color space if needed
input_luma_im = PrepareImage(input_im);
orig_luma_im = PrepareImage(orig_im);

if strcmp(deg_model,'Downscale')
    % upscale using bicubic
    input_im = imresize(input_im,scale,'bicubic');
    input_im = input_im(1:size(orig_im,1), 1:size(orig_im,2), :); 
end
% fprintf(' Done.\n');
psnr_input = ComputePSNR(orig_im, input_im);
deg_im = input_im;

%% minimize the Laplacian regularization functional

switch reg_function
    case 'Fixed-Point'      
        
        switch deg_model
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
        
        switch deg_model
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
        tStart = cputime;
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
        
        switch deg_model
            case 'UniformBlur'
                params_sd = GetUniformDeblurSDParams(light_mode);
            case 'GaussianBlur'
                params_sd = GetGaussianDeblurSDParams(light_mode);
            case 'Downscale'
                params_sd = GetSuperResSDParams(light_mode);
            otherwise
                error('Degradation model is not defined');
        end
        tStart = cputime;
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
% For analysis - convert original frame to image 
orig_im = double(orig_im);
output_im = double(output_im);

tEnd = cputime - tStart;

ssim_calc = ssim(orig_im, output_im);
ssim_out = ssim_calc;

abs_difr_im = imabsdiff(orig_im, output_im);
abs_difr_mean = mean(abs_difr_im,'all');
abs_diff = abs_difr_mean;

RMSE_calc =  sqrt(mean(mean((output_im - orig_im).^2)));
RMSE_val = RMSE_calc;

RMSE_noisy_calc =  sqrt(mean(mean((input_im - orig_im).^2)));
RMSE_noisy_val = RMSE_noisy_calc;


