%% Initialisation
clc
clear all
close all

cd C:\Users\saifs\Desktop\LIDARTesting
prenet = denoisingNetwork('DnCNN')
load net
load net2
load net3
load net4
% cd F:\
load net5

cd C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results
load Face_Depth_Reflectivity_Clean.mat DepthImage
I = DepthImage
D_HR = DepthImage

% I = imread('C:\Users\saifs\Desktop\LIDARTesting\AudiR8300T.png')
% I = double(rgb2gray(I))

[row, col] = size(I)
Iref = I./255
noisyI = max(0,  double(I) + 20*randn(size(I)));
noisyI = noisyI./255
denoisedI1 = denoiseImage(noisyI, net)
denoisedI2 = denoiseImage(noisyI, net2)
denoisedI3 = denoiseImage(noisyI, net3)
denoisedI4 = denoiseImage(noisyI, net4)

RMSE1 = sqrt(mean(mean((Iref  - denoisedI1).^2)));
RMSE2 = sqrt(mean(mean((Iref  - denoisedI2).^2)));
RMSE3 = sqrt(mean(mean((Iref  - denoisedI3).^2)));
RMSE4 = sqrt(mean(mean((Iref  - denoisedI4).^2)));

denoisedImage = cell(1, 20);
%% Computation of Denoised and Filtered Images
parfor loop = 1:20
    sigma(loop) = 5*loop;
    Inoisy{loop} = max(0,  double(I) + sigma(loop)*randn(size(I)));
    Inoisy{loop} = Inoisy{loop}./255;
    denoisedImage{loop} = denoiseImage(Inoisy{loop}, net4);
    denoisedIPre{loop} = denoiseImage(Inoisy{loop}, prenet);
    denoisedIUnf{loop} = denoisedImage{loop}
    denoisedIPreUnf{loop} = denoisedIPre{loop}
    
    fire = 0;
for i = 2:1:row - 1
    for j = 2:1:col - 1
       Ext = 0;
       ExtPre = 0;
       North = denoisedImage{loop}(i-1, j);
       West = denoisedImage{loop}(i, j-1);
       East = denoisedImage{loop}(i, j+1);
       South = denoisedImage{loop}(i+1, j);
       
       NorthPre = denoisedIPre{loop}(i-1, j);
       WestPre = denoisedIPre{loop}(i, j-1);
       EastPre = denoisedIPre{loop}(i, j+1);
       SouthPre = denoisedIPre{loop}(i+1, j);
        
       diffN = abs(denoisedImage{loop}(i,j) - North);
       diffW = abs(denoisedImage{loop}(i,j) - West);
       diffE = abs(denoisedImage{loop}(i,j) - East);
       diffS = abs(denoisedImage{loop}(i,j) - South);
       
       diffNPre = abs(denoisedIPre{loop}(i,j) - NorthPre);
       diffWPre = abs(denoisedIPre{loop}(i,j) - WestPre);
       diffEPre = abs(denoisedIPre{loop}(i,j) - EastPre);
       diffSPre = abs(denoisedIPre{loop}(i,j) - SouthPre);
       
       Threshold = 0.08;
       if(diffN > Threshold)
           Ext = Ext + 1;
       end
        if(diffE > Threshold)
           Ext = Ext + 1;
        end
        if(diffW > Threshold)
           Ext = Ext + 1;
        end
        if(diffS > Threshold)
           Ext = Ext + 1;
        end
       
        if(Ext >= 1)
            denoisedImage{loop}(i, j) = NaN;
            fire = fire + 1;
        end

        if(diffNPre > Threshold)
           ExtPre = ExtPre + 1;
       end
        if(diffEPre > Threshold)
           ExtPre = ExtPre + 1;
        end
        if(diffWPre > Threshold)
           ExtPre = ExtPre + 1;
        end
        if(diffSPre > Threshold)
           ExtPre = ExtPre + 1;
        end
       
        if(Ext >= 1)
            denoisedImage{loop}(i, j) = NaN;
        end
        
        if(ExtPre >= 1)
            denoisedIPre{loop}(i, j) = NaN;
        end
    end
end

    denoisedImage{loop} = denoisedImage{loop}.*255
    denoisedIUnf{loop} = denoisedIUnf{loop}.*255
    Inoisy{loop} = Inoisy{loop}.*255
    denoisedIPreUnf{loop} = denoisedIPreUnf{loop}.*255
    denoisedIPre{loop} = denoisedIPre{loop}.*255
    
    SNR_data(loop) = 10*log10(std(I(:))/sigma(loop)); %% modified by AH
    RMSE_Denoised(loop) = sqrt(mean(mean((I  - denoisedImage{loop}).^2), 'omitnan'));
    RMSE_DenoisedFilt(loop) = sqrt(mean(mean((I  - denoisedIUnf{loop}).^2)));      %RMSE of UNFILTERED TRAINED denoised Image
    RMSE_noisy(loop) = sqrt(mean(mean((I - Inoisy{loop}).^2)));
    RMSE_PreNetUnf(loop) = sqrt(mean(mean((I - denoisedIPreUnf{loop}).^2)));
    RMSE_PreNetFilt(loop) = sqrt(mean(mean((I - denoisedIPre{loop}).^2), 'omitnan'));
end

   %% Displaying Network Comparison
figure
subplot(3,2, [1 2])
imshow(uint8(noisyI.*255-300))
title('Noisy Image')
subplot(3, 2, 3)
imshow(uint8(denoisedI1.*255-300))
title({'Network 1', 'Parallel CPU, 3 Epochs', 'Dataset Size: 23 Images'})
subplot(3, 2, 4)
imshow(uint8(denoisedI2.*255-300))
title({'Network 2', 'Parallel CPU, 5 Epochs', 'Dataset Size: 23 Images'})
subplot(3, 2, 5)
imshow(uint8(denoisedI3.*255-300))
title({'Network 3', 'Parallel CPU, 3 Epochs', 'Dataset Size: 100 Images'})
subplot(3, 2, 6)
imshow(uint8(denoisedI4.*255-300))
title({'Network 4', 'Single GPU, 20 Epochs', 'Dataset Size: 200 Images'})

pause
close all
%% Displaying Ref/Denoised Comparison
roi = [26 36 20 20]
figure
montage({imcrop(uint8(I-300), roi), imcrop(uint8(denoisedI4.*255-300), roi)})
title('Reference Image - Object Edge (Left) | Denoised Image - Object Edge (Right)', 'FontSize', 26)
I = I.*255
noisyI = noisyI.*255

figure
    x = kron(ones(row,1),(1:col)')
    y = reshape(I(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))
    subplot(1, 3, 1)
    scatter3(x, y, z, 50 ,'.'),
    title('Reference Point Cloud')
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(noisyI(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))
    subplot(1, 3, 2)
    scatter3(x, y, z, 50 ,'.'),
    title('Noisy Point Cloud')
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedI4(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))
    subplot(1, 3, 3)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised Point Cloud')
    
    pause
    close all
    %% Choosing Image and showing data
    dataset = input('Select Image to Analyze\n')
        
    x = kron(ones(row,1),(1:col)')
    y = reshape(Inoisy{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,1)
    scatter3(x, y, z, 50 ,'.'),
    title(strcat('Noisy, \sigma = ',  string(5*dataset)), 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedIUnf{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,2)
    scatter3(x, y, z, 50 ,'.'),
    title(strcat('Denoised PC, \sigma = ', string(5*dataset)), 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedImage{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,3)
    scatter3(x, y, z, 50 ,'.'),
    title(strcat('Denoised PC - Outliers Removed, \sigma = ', string(5*dataset)), 'FontSize', 26)
    
    figure
    montage({uint8(Inoisy{dataset}-300), uint8(denoisedIUnf{dataset}-300), uint8(denoisedImage{dataset}-300)}, 'Size', [1 3])
    title('Noisy Image (Left), Denoised Image (Center), Outliers Removed (Right)')
    
    pause 
    close all
    %% Analytical Results
    figure
    title('Network Comparison (With Filtering)')
    hold on
    plot(SNR_data, RMSE_Denoised, 'r')
    plot(SNR_data, RMSE_noisy, 'k')
    plot(SNR_data, RMSE_DenoisedFilt, 'r--')
    plot(SNR_data, RMSE_PreNetUnf, 'b')
    plot(SNR_data, RMSE_PreNetFilt, 'b--')
    hold off
    xlabel('SNR');
    ylabel('RMSE');
    legend('RMSE(Denoised)','RMSE(Noisy)','RMSE (Denoised + Filtered)', 'RMSE(Pretrained)', 'RMSE(Pretrained + Filtered)')

    pause 
    close all
    %% Pretrained vs Trained Comparison
    figure
    montage({uint8(Inoisy{dataset} - 300), uint8(denoisedIUnf{dataset} - 300), uint8(denoisedIPreUnf{dataset} - 300)}, 'Size', [1 3])
    title(strcat('Noisy Image (Left) | Denoised Image - Trained Network (Center) | Denoised Image - Pretrained Network (Right)', ' | \sigma = ', string(5*dataset)), 'FontSize', 24)
    
    figure
    subplot(1,3,1)
    x = kron(ones(row,1),(1:col)')
    y = reshape(Inoisy{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,1)
    scatter3(x, y, z, 50 ,'.'),
    title(strcat('Noisy, \sigma = ',  string(5*dataset)), 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedIUnf{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,2)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC - Trained', 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedIPreUnf{dataset}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,3)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC - Pretrained', 'FontSize', 26)
    
    pause 
    close all
    
    %% SR Computation
    
    clc
    clear all
    load('trainedVDSR-Epoch-100-ScaleFactors-234.mat');

    cd C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results
    load Face_Depth_Reflectivity_Clean.mat DepthImage


    % D_HR = imread("C:\Users\saifs\Downloads\RWMH-Instance.png")
    D_HR = DepthImage
    D_HR = D_HR - 300
    % D_HR = imcomplement(D_HR)
    D_HR = uint8(D_HR)
    % D_HR = rgb2gray(D_HR)
    [row, col] = size(D_HR)

    Ireference = D_HR


    for loop = 1:5
    D_HR = double(D_HR)
    scaleFactor{loop} = loop*0.1;
    Ilowres{loop} = imresize(Ireference,scaleFactor{loop},'bicubic');
    IlowresC{loop} = cat(3, Ilowres{loop}, Ilowres{loop}, Ilowres{loop});
    Iycbcr{loop} = rgb2ycbcr(IlowresC{loop});
    Iy{loop} = Iycbcr{loop}(:,:,1);
    Icb{loop} = Iycbcr{loop}(:,:,2);
    Icr{loop} = Iycbcr{loop}(:,:,3);

    [nrows,ncols,np] = size(Ireference);
    Ibicubic{loop} = imresize(IlowresC{loop},[nrows ncols],'bicubic');

    Iy_bicubic{loop} = double(imresize(Iy{loop},[nrows ncols],'bicubic'));
    Icb_bicubic{loop} = imresize(Icb{loop},[nrows ncols],'bicubic');
    Icr_bicubic{loop} = imresize(Icr{loop},[nrows ncols],'bicubic');

    Iresidual{loop} = activations(net,Iy_bicubic{loop},41, 'ExecutionEnvironment', 'auto');
    Iresidual{loop} = double(Iresidual{loop});

    Isr{loop} = Iy_bicubic{loop} + Iresidual{loop};

    [rowLR{loop}, colLR{loop}] = size(rgb2gray(IlowresC{loop}))             
    Ivdsr{loop} = ycbcr2rgb(cat(3,Isr{loop},Icb_bicubic{loop},Icr_bicubic{loop}));


    Ivdsr{loop} = double(rgb2gray(Ivdsr{loop}))



        x = kron(ones(row,1),(1:col)')
        y = reshape(Ivdsr{loop}(end:-1:1,:)',row*col,1)'
        z = kron((1:row)',ones(col,1))

        subplot(2,3,loop)
        scatter3(x, y, z, 50 ,'.'),


        IvdsrUnf{loop} = Ivdsr{loop}
        %%
        for i = 2:1:row - 1
        for j = 2:1:col - 1
           Ext = 0;
           ExtPre = 0;
           North = Ivdsr{loop}(i-1, j);
           West = Ivdsr{loop}(i, j-1);
           East = Ivdsr{loop}(i, j+1);
           South = Ivdsr{loop}(i+1, j);

           diffN = abs(Ivdsr{loop}(i,j) - North);
           diffW = abs(Ivdsr{loop}(i,j) - West);
           diffE = abs(Ivdsr{loop}(i,j) - East);
           diffS = abs(Ivdsr{loop}(i,j) - South);


           if(diffN > 5)
               Ext = Ext + 1;
           end
            if(diffE > 5)
               Ext = Ext + 1;
            end
            if(diffW > 5)
               Ext = Ext + 1;
            end
            if(diffS > 5)
               Ext = Ext + 1;
            end
            
            if(Ext >= 1)
                Ivdsr{loop}(i, j) = NaN;
            end


        end
        end


        x = kron(ones(row,1),(1:col)')
        y = reshape(Ivdsr{loop}(end:-1:1,:)',row*col,1)'
        z = kron((1:row)',ones(col,1))

        subplot(2,3,loop)
        scatter3(x, y, z, 50 ,'.'),
        title(strcat('Upscaled image - Sampling Rate = ', string(scaleFactor{loop}*100), '%'), 'FontSize', 26)

        LRRescaled{loop} = double(imresize(Ilowres{loop}, [row col]))
        RMSE_VDSR(loop) = sqrt(mean(mean(((D_HR  - Ivdsr{loop}).^2),  'omitnan')));
        RMSE_LR(loop) = sqrt(mean(mean(((D_HR  - LRRescaled{loop}).^2))));
        RMSE_VDSRUnfilt(loop) = sqrt(mean(mean(((D_HR  - IvdsrUnf{loop}).^2))));
        ssimval(loop) = ssim(IvdsrUnf{loop}, D_HR)
    end
    
    
%% LR/HR Comparison

        x = kron(ones(row,1),(1:col)')
        y = reshape(D_HR(end:-1:1,:)',row*col,1)'
        z = kron((1:row)',ones(col,1))

        subplot(1,4,1)
        scatter3(x, y, z, 50 ,'.'),
        title('Reference PC', 'FontSize', 26)
                
        x = kron(ones(rowLR{3},1),(1:colLR{3})')
        y = reshape(Ilowres{3}(end:-1:1,:)',rowLR{3}*colLR{3},1)'
        z = kron((1:rowLR{3})',ones(colLR{3},1))

        subplot(1,4,2)
        scatter3(x, y, z, 50 ,'.'),
        title('Low-Resolution PC', 'FontSize', 26)
        
        x = kron(ones(row,1),(1:col)')
        y = reshape(IvdsrUnf{3}(end:-1:1,:)',row*col,1)'
        z = kron((1:row)',ones(col,1))

        subplot(1,4,3)
        scatter3(x, y, z, 50 ,'.'),
        title('Super-Resolved PC', 'FontSize', 26)
        
        x = kron(ones(row,1),(1:col)')
        y = reshape(Ivdsr{3}(end:-1:1,:)',row*col,1)'
        z = kron((1:row)',ones(col,1))

        subplot(1,4,4)
        scatter3(x, y, z, 50 ,'.'),
        title('Super-Resolved PC (Outliers Removed)', 'FontSize', 26)
        
    %% Analytical Results
    figure
    subplot(1,2,1)
    plot(RMSE_LR, 'k')
    title('RMSE by Index', 'FontSize', 26)
    hold on
    plot(RMSE_VDSR, 'r--')
    plot(RMSE_VDSRUnfilt, 'r')
    xlabel('Index')
    ylabel('RMSE')
    legend('RMSE - Low Resolution', 'RMSE - Super-Resolved + Filtered', 'RMSE - Super-Resolved')
    subplot(1,2,2)
    plot(ssimval)
    title('SSIM by Index', 'FontSize', 26)
    xlabel('Index')
    ylabel('SSIM')
    legend('SSIM - Super-Resolved')
    
