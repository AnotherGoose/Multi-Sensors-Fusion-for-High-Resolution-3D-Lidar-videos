%% Result Anaylsis Function
%....
%....

%% Load Data
% Noise Levels Measured: 1/255, sqrt(2), 4, 8 
x_noise = [1/255,1,2,3,4,5,6,7,8,9,10];

data_files = {'noiseS.mat','noise1.mat','noise2.mat','noise3.mat','noise4.mat','noise5.mat','noise6.mat','noise7.mat','noise8.mat','noise9.mat','noise10.mat'};

for i = 1:numel(data_files)
    load(data_files{i})
    % mat_DnCNN
    ADMM_mat_DnCNN_psnrOut(i) = [ADMM_mat_DnCNN.psnrOut];
    ADMM_mat_DnCNN_ssim(i) = [ADMM_mat_DnCNN.ssim];
    ADMM_mat_DnCNN_absDiff(i) = [ADMM_mat_DnCNN.abs_diffrence];
    ADMM_mat_DnCNN_compt(i) = [ADMM_mat_DnCNN.time]; 
    ADMM_mat_DnCNN_rsme(i) = [ADMM_mat_DnCNN.RMSE];

    
    % TNRD
    ADMM_tnrd_psnrOut(i) = [ADMM_tnrd.psnrOut];
    ADMM_tnrd_ssim(i) = [ADMM_tnrd.ssim];
    ADMM_tnrd_absDiff(i) = [ADMM_tnrd.abs_diffrence];
    ADMM_tnrd_compt(i) = [ADMM_tnrd.time]; 
    ADMM_tnrd_rsme(i) = [ADMM_tnrd.RMSE];

    
    % Median 3x3
    ADMM_Med3_psnrOut(i) = [ADMM_Med3.psnrOut];
    ADMM_Med3_ssim(i) = [ADMM_Med3.ssim];
    ADMM_Med3_absDiff(i) = [ADMM_Med3.abs_diffrence];
    ADMM_Med3_compt(i) = [ADMM_Med3.time];
    ADMM_Med3_rsme(i) = [ADMM_Med3.RMSE];

    
    % Wiener 
    ADMM_Wiener_psnrOut(i) = [ADMM_Wiener.psnrOut];
    ADMM_Wiener_ssim(i) = [ADMM_Wiener.ssim];
    ADMM_Wiener_absDiff(i) = [ADMM_Wiener.abs_diffrence];
    ADMM_Wiener_compt(i) = [ADMM_Wiener.time];
    FP_Wiener_rsme(i) = [ADMM_Wiener.RMSE];

    
    % FP
    % mat_DnCNN
    FP_mat_DnCNN_psnrOut(i) = [FP_mat_DnCNN.psnrOut];
    FP_mat_DnCNN_ssim(i) = [FP_mat_DnCNN.ssim];
    FP_mat_DnCNN_absDiff(i) = [FP_mat_DnCNN.abs_diffrence];
    FP_mat_DnCNN_compt(i) = [FP_mat_DnCNN.time];
    FP_mat_DnCNN_rsme(i) = [FP_mat_DnCNN.RMSE];

    
    % TNRD
    FP_tnrd_psnrOut(i) = [FP_tnrd.psnrOut];
    FP_tnrd_ssim(i) = [FP_tnrd.ssim];
    FP_tnrd_absDiff(i) = [FP_tnrd.abs_diffrence];
    FP_tnrd_compt(i) = [FP_tnrd.time];
    FP_tnrd_rsme(i) = [FP_tnrd.RMSE];

    
    % Median 3x3
    FP_Med3_psnrOut(i) = [FP_Med3.psnrOut];
    FP_Med3_ssim(i) = [FP_Med3.ssim];
    FP_Med3_absDiff(i) = [FP_Med3.abs_diffrence];
    FP_Med3_compt(i) = [FP_Med3.time];    
    FP_Med3_rsme(i) = [FP_Med3.RMSE];

    
    % Wiener 
    FP_Wiener_psnrOut(i) = [FP_Wiener.psnrOut];
    FP_Wiener_ssim(i) = [FP_Wiener.ssim];
    FP_Wiener_absDiff(i) = [FP_Wiener.abs_diffrence];
    FP_Wiener_compt(i) = [FP_Wiener.time];
    FP_Wiener_rsme(i) = [FP_Wiener.RMSE];
    
end
%% Analysis for best method
% - Search for best value for each comp methods
%   - Return for each noise level
%   - Return overall best
%   - Have to search through each (i) for each noise level seperately to be
%   able to compare 
for a = 1:i
% SSIM
% - Scale of 0 - 1 
% - Higher SSIM = 1 - Greater Structural Similarity
% - Lower SSIM = 0 - Lower Structural Similarity 
    ssim_val = [ADMM_mat_DnCNN_ssim(a), ADMM_tnrd_ssim(a), ADMM_Med3_ssim(a), ADMM_Wiener_ssim(a),...
                    FP_mat_DnCNN_ssim(a), FP_tnrd_ssim(a), FP_Med3_ssim(a), FP_Wiener_ssim(a)];
    ssim_val_name = {'ADMM_mat_DnCNN_ssim','ADMM_tnrd_ssim','ADMM_Med3_ssim','ADMM_Wiener_ssim',...
                    'FP_mat_DnCNN_ssim','FP_tnrd_ssim','FP_Med3_ssim','FP_Wiener_ssim'};    
        
    ssim_maxVal = max(ssim_val,[],'all');
    best_ssimPosition = find(ssim_val==ssim_maxVal);
    best_ssim(a) = ssim_val_name(best_ssimPosition);
    ssim_val = 0;
    
% PSNR
% - Higher PSNR = Greater Result
% - Lower PSNR = Lower Quality Result
    psnr_val = [ADMM_mat_DnCNN_psnrOut(a), ADMM_tnrd_psnrOut(a), ADMM_Med3_psnrOut(a), ADMM_Wiener_psnrOut(a),...
                    FP_mat_DnCNN_psnrOut(a), FP_tnrd_psnrOut(a), FP_Med3_psnrOut(a), FP_Wiener_psnrOut(a)];
    psnr_val_name = {'ADMM_mat_DnCNN_psnr','ADMM_tnrd_psnr','ADMM_Med3_psnr','ADMM_Wiener_psnr',...
                    'FP_mat_DnCNN_psnr','FP_tnrd_psnr','FP_Med3_psnr','FP_Wiener_psnr'};    
        
    psnr_maxVal = max(psnr_val,[],'all');
    best_psnrPosition = find(psnr_val==psnr_maxVal);
    best_psnr(a) = psnr_val_name(best_psnrPosition);
    psnr_val = 0;

% Absolute Difference
% - Higher AbsDiff = Greater difference between 2 images
% - Lower AbsDiff = Low difference between 2 images
    absDifr_val = [ADMM_mat_DnCNN_absDiff(a), ADMM_tnrd_absDiff(a), ADMM_Med3_absDiff(a), ADMM_Wiener_absDiff(a),...
                    FP_mat_DnCNN_absDiff(a), FP_tnrd_absDiff(a), FP_Med3_absDiff(a), FP_Wiener_absDiff(a)];
    absDifr_val_name = {'ADMM_mat_DnCNN_absDifr','ADMM_tnrd_absDifr','ADMM_Med3_absDifr','ADMM_Wiener_absDifr',...
                    'FP_mat_DnCNN_absDifr','FP_tnrd_absDifr','FP_Med3_absDifr','FP_Wiener_absDifr'};    
        
    absDifr_maxVal = min(absDifr_val,[],'all');
    best_absDifrPosition = find(absDifr_val==absDifr_maxVal);
    best_absDifr(a) = absDifr_val_name(best_absDifrPosition);
    absDifr_val = 0;
end

% Determine which was best overall for each comparison method
ssim_m = strtrim(mode(char(best_ssim)));
psnr_m = strtrim(mode(char(best_psnr)));
absDifr_m = strtrim(mode(char(best_absDifr)));

%% Result Plot

% Noise Level against PSNR
figure()
plot(x_noise,ADMM_tnrd_psnrOut,'m', 'DisplayName','ADMM TNRD')
xlabel('Noise Level'),
ylabel('PSNR'),
title('PSNR Comparison of Denoising Algorithms With RED')
grid on
legend
hold on
plot(x_noise, ADMM_mat_DnCNN_psnrOut,'g', 'DisplayName','ADMM DnCNN')
hold on
plot(x_noise, ADMM_Med3_psnrOut,'b', 'DisplayName','ADMM MedianF 3x3')
hold on
plot(x_noise, ADMM_Wiener_psnrOut,'r', 'DisplayName','ADMM Wiener Filter')
hold on
plot(x_noise, FP_mat_DnCNN_psnrOut,'g--', 'DisplayName','FP DnCNN')
hold on
plot(x_noise, FP_Med3_psnrOut,'b--', 'DisplayName','FP MedianF 3x3')
hold on
plot(x_noise, FP_Wiener_psnrOut,'r--', 'DisplayName','FP Wiener Filter')
hold on
plot(x_noise, FP_tnrd_psnrOut,'m--', 'DisplayName','FP TNRD')
hold off

% Noise Level against ssim
figure()
plot(x_noise,ADMM_tnrd_ssim,'m', 'DisplayName','ADMM TNRD')
xlabel('Noise Level'),
ylabel('SSIM'),
title('SSIM Comparison of Denoising Algorithms With RED')
grid on
legend
hold on
plot(x_noise, ADMM_mat_DnCNN_ssim,'g', 'DisplayName','ADMM DnCNN')
hold on
plot(x_noise, ADMM_Med3_ssim,'b', 'DisplayName','ADMM MedianF 3x3')
hold on
plot(x_noise, ADMM_Wiener_ssim,'r', 'DisplayName','ADMM Wiener Filter')
hold on
plot(x_noise, FP_mat_DnCNN_ssim,'g--', 'DisplayName','FP DnCNN')
hold on
plot(x_noise, FP_Med3_ssim,'b--', 'DisplayName','FP MedianF 3x3')
hold on
plot(x_noise, FP_Wiener_ssim,'r--', 'DisplayName','FP Wiener Filter')
hold on
plot(x_noise, FP_tnrd_ssim,'m--', 'DisplayName','FP TNRD')
hold off

% Noise Level against Absolute Difference
figure()
plot(x_noise,ADMM_tnrd_absDiff,'m', 'DisplayName','ADMM TNRD')
xlabel('Noise Level'),
ylabel('Absolute Difference'),
title('Absolute Difference Comparison of Denoising Algorithms With RED')
grid on
legend
hold on
plot(x_noise, ADMM_mat_DnCNN_absDiff,'g', 'DisplayName','ADMM DnCNN')
hold on
plot(x_noise, ADMM_Med3_absDiff,'b', 'DisplayName','ADMM MedianF 3x3')
hold on
plot(x_noise, ADMM_Wiener_absDiff,'r', 'DisplayName','ADMM Wiener Filter')
hold on
plot(x_noise, FP_mat_DnCNN_absDiff,'g--', 'DisplayName','FP DnCNN')
hold on
plot(x_noise, FP_Med3_absDiff,'b--', 'DisplayName','FP MedianF 3x3')
hold on
plot(x_noise, FP_Wiener_absDiff,'r--', 'DisplayName','FP Wiener Filter')
hold on
plot(x_noise, FP_tnrd_absDiff,'m--', 'DisplayName','FP TNRD')
hold off

% Noise Level against Computation Time
figure()
plot(x_noise,ADMM_tnrd_compt,'m', 'DisplayName','ADMM TNRD')
xlabel('Noise Level'),
ylabel('Computation Time (s)'),
title('Computation Time Comparison of Denoising Algorithms With RED')
grid on
legend
hold on
plot(x_noise, ADMM_mat_DnCNN_compt,'g', 'DisplayName','ADMM DnCNN') 
hold on
plot(x_noise, ADMM_Med3_compt,'b', 'DisplayName','ADMM MedianF 3x3')
hold on
plot(x_noise, ADMM_Wiener_compt,'r', 'DisplayName','ADMM Wiener Filter')
hold on
plot(x_noise, FP_mat_DnCNN_compt,'g--', 'DisplayName','FP DnCNN')
hold on
plot(x_noise, FP_Med3_compt,'b--', 'DisplayName','FP MedianF 3x3')
hold on
plot(x_noise, FP_Wiener_compt,'r--', 'DisplayName','FP Wiener Filter')
hold on
plot(x_noise, FP_tnrd_compt,'m--', 'DisplayName','FP TNRD')
hold off
