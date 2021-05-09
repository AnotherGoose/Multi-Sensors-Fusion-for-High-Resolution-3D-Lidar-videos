%% Controller Script
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

%% Read and Perform Actions on Data

% Load Input Data
input_data = 'Jk_LowNoise.mp4';
%input_data = 'barbara.tif';

% Choose whether to analyise one frame or the entire video: 'image','frame','video'
anaylsis_md = 'frame';

% choose the secenrio: 'UniformBlur', 'GaussianBlur', or 'Downscale'
deg_model = 'GaussianBlur';

% Regularization functional: 'ADMM','Fixed-Point','Steepest-Descent
% Denoising engine to be used: 'tnrd','mat_DnCNN','Med3','Wnr'

%% Run Scenerios
reg_function = 'Fixed-Point';
noise_level = 1/255;

% FP TNRD
denoising_eng = 'tnrd';
fprintf('Restoring using RED: %s %s method\n',reg_function,denoising_eng);
[psnr_input,psnr_out,tEnd,ssim_out,abs_diff,output_im,RMSE_val,RMSE_noisy_val] = action_function(input_data,anaylsis_md,noise_level,reg_function,denoising_eng,deg_model,light_mode);
noiseS.FP_tnrd = struct('psnrIn',{psnr_input},'psnrOut',{psnr_out},'time',{tEnd},'ssim',{ssim_out},'abs_diffrence',{abs_diff},'rmse',{RMSE_val},'rmse_noisy',{RMSE_noisy_val});
%{ 
noise_level = 1;

% FP TNRD
denoising_eng = 'tnrd';
fprintf('Restoring using RED: %s %s method\n',reg_function,denoising_eng);
[psnr_input,psnr_out,tEnd,ssim_out,abs_diff,output_im,RMSE_val,RMSE_noisy_val] = action_function(input_data,anaylsis_md,noise_level,reg_function,denoising_eng,deg_model,light_mode);
noise1.FP_tnrd = struct('psnrIn',{psnr_input},'psnrOut',{psnr_out},'time',{tEnd},'ssim',{ssim_out},'abs_diffrence',{abs_diff},'rmse',{RMSE_val},'rmse_noisy',{RMSE_noisy_val});

%}
