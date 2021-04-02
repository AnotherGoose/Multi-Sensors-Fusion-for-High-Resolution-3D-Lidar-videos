clear all
close all
clc
net = denoisingNetwork('DnCNN');

tick = 0

cd C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results
addpath 'C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results'
addpath 'C:\Users\saifs\Desktop\LIDARTesting\TrainingDataset\comped'
load Face_Depth_Reflectivity_Clean.mat DepthImage
I = DepthImage;
I = max(I, 0)
% I = imread('in_00_160201_141702_depth_vi.png')
I = double(I)
Ext = 0;
% I = double(I - 228)
Iref = I/255;%% modified by AH
[row, col] = size(I);
%% Change this to manually generated noise
    sigma = 50
%     Inoisy = max(0,  double(I) + sigma*randn(size(I)));
%     Inoisy = Inoisy./255;
% fire = 0;
    cd C:\Users\saifs\Desktop\LIDARTesting
    load net4
    prenet = denoisingNetwork('DnCNN')
%     denoisedI = denoiseImage(Inoisy, net4);          %Between 0 and 1

    parfor loop = 1:20
    sigma(loop) = 5*loop;
    Inoisy{loop} = max(0,  double(I) + sigma(loop)*randn(size(I)));
    Inoisy{loop} = Inoisy{loop}./255;
    denoisedI{loop} = denoiseImage(Inoisy{loop}, net4);
    denoisedIPre{loop} = denoiseImage(Inoisy{loop}, prenet);
    test{loop} = denoisedI{loop}
    testPre{loop} = denoisedIPre{loop}
    Ext = 0;
    fire = 0;
for i = 2:1:row - 1
    for j = 2:1:col - 1
%        Ext = 0;
       North = denoisedI{loop}(i-1, j);
       West = denoisedI{loop}(i, j-1);
       East = denoisedI{loop}(i, j+1);
       South = denoisedI{loop}(i+1, j);
       NorthWest = denoisedI{loop}(i-1, j-1);
       NorthEast = denoisedI{loop}(i-1, j+1);
       SouthWest = denoisedI{loop}(i+1, j-1);
       SouthEast = denoisedI{loop}(i+1, j+1);
       
        
       diffN = abs(denoisedI{loop}(i,j) - North);
       diffW = abs(denoisedI{loop}(i,j) - West);
       diffE = abs(denoisedI{loop}(i,j) - East);
       diffS = abs(denoisedI{loop}(i,j) - South);
       diffNW = abs(denoisedI{loop}(i,j) - NorthWest);
       diffNE = abs(denoisedI{loop}(i,j) - NorthEast);
       diffSW = abs(denoisedI{loop}(i,j) - SouthWest);
       diffSE = abs(denoisedI{loop}(i,j) - SouthEast);
       
       diffAvg = (diffN + diffW + diffE + diffS)/4;
       
%        if(diffAvg > 0.1)
%            Ext = Ext + 1;
%            denoisedI{loop}(i, j) = NaN;
%            fire = fire + 1;
%        end
%        
%         if(Ext >= 2)
%             denoisedI{loop}(i, j) = NaN;
%             fire = fire + 1;
%         end
%         
       
%         if(Ext >= 4)
%             denoisedI{loop}(i, j) = NaN;
%             denoisedI{loop}(i, j) = median(denoisedI{loop}(i-1:i+1,j-1:j+1), 'All','includenan');
        end
        
end

for i = 2:row - 1
    for j = 2:col - 1
 if(diffAvg > 0.09775)
           Ext = Ext + 1;
           denoisedI{loop}(i, j) = NaN;
           fire = fire + 1;
 end
    end
end
       
MedianTest{loop} = denoisedI{loop}
for i = 1:row
    for j = 1:col
    if any(isnan(denoisedI{loop}(i, j)))
        MedianTest{loop}(i, j) = nanmedian(denoisedI{loop}(i-1:i+1,j-1:j+1), 'All');
        tick = tick  + 1;
    end
    end
end
    %Correct to only consider odd numbers of scanned pixels
    
    
    RMSE_MedianTest(loop) = sqrt(mean(mean((Iref  - MedianTest{loop}).^2), 'omitnan'));
    
    SNR_data(loop) = 10*log10(std(I(:))/sigma(loop)); %% modified by AH
    RMSE_normalizedDataTest(loop) = sqrt(mean(mean((Iref  - test{loop}).^2)));      %RMSE of UNFILTERED TRAINED denoised Image
    RMSE_normalizedData(loop) = sqrt(mean(mean((Iref  - denoisedI{loop}).^2), 'omitnan'));  %RMSE of FILTERED TRAINED denoised Image
    RMSE_noisy(loop) = sqrt(mean(mean((Iref - Inoisy{loop}).^2)));

    RMSE_normalizedDataPre(loop) = sqrt(mean(mean((Iref  - testPre{loop}).^2)));    %RMSE of UNFILTERED PRETRAINED denoised Image
    RMSE_normalizedDataPreFilt(loop) = sqrt(mean(mean((Iref  - denoisedIPre{loop}).^2), 'omitnan'));    %RMSE of FILTERED PRETRAINED denoised Image
% test{loop} = test{loop}.*255
% denoisedI{loop} = denoisedI{loop}.*255
% Inoisy{loop} = Inoisy{loop}.*255
    x = kron(ones(row,1),(1:col)')
    y = reshape(test{loop}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(2,10,loop)
    scatter3(x, y, z, 50 ,'.'),
    
    
    denoisedI{loop} = denoisedI{loop}.*255
    MedianTest{loop} = MedianTest{loop}.*255
     
    test{loop} = test{loop}.*255
    end
    
figure
plot(SNR_data,RMSE_normalizedDataTest, 'r')
title('Network Comparison (With Filtering)')
hold on
plot(SNR_data, RMSE_normalizedData, 'r--')
plot(SNR_data, RMSE_noisy, 'k')
plot(SNR_data, RMSE_normalizedDataPre, 'b')
plot(SNR_data, RMSE_normalizedDataPreFilt, 'b--')
hold off
xlabel('SNR');
ylabel('RMSE');
legend('RMSE(Denoised)','RMSE(Denoised + Filtered)','RMSE (Noisy)', 'Pretrained', 'Pretrained + Filtered')

figure
plot(SNR_data, RMSE_MedianTest)
hold on
plot(SNR_data, RMSE_normalizedData)
legend('Median Filtered', 'Unfiltered')
xlabel('SNR')
ylabel('RMSE')

figure
plot(SNR_data,RMSE_normalizedDataTest, 'r')
title('Network Comparison (With Filtering)')
hold on
plot(SNR_data, RMSE_noisy, 'k')
plot(SNR_data, RMSE_normalizedDataPre, 'b')
hold off
title('Simplified Network Comparison')
xlabel('SNR');
ylabel('RMSE');
legend('RMSE(Denoised)','RMSE (Noisy)', 'Pretrained')

% figure
% plot(sigma,RMSE_normalizedDataTest);
% hold on
% plot(sigma,RMSE_normalizedDataPre);
% plot(sigma, RMSE_noisy)
% legend('Denoised Images', 'Denoised Images (Pretrained)', 'Noisy Images')
% xlabel('\sigma of Noise')
% ylabel('RMSE')
% 
figure

    x = kron(ones(row,1),(1:col)')
    y = reshape(Inoisy{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,1)
    scatter3(x, y, z, 50 ,'.'),
    title('Noisy PC (\sigma = 20)')
     
    x = kron(ones(row,1),(1:col)')
    y = reshape(Inoisy{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,2)
    scatter3(x, y, z, 50 ,'.'),
    title('Noisy PC (\sigma = 20)')
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(testPre{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,3)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC (Pretrained)')
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(test{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,4)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC (Trained)')
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedIPre{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,5)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC (Pretrained + Filtered)')
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedI{14}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(3,2,6)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised PC (Trained + Filtered)')
    
    
    
    
    
    
    
    %%
    close all
    
        
    x = kron(ones(row,1),(1:col)')
    y = reshape(test{12}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,1)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised Point Cloud', 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(denoisedI{12}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,2)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised Point Cloud - Outliers Removed (\sigma = 60)', 'FontSize', 26)
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(MedianTest{12}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,3,3)
    scatter3(x, y, z, 50 ,'.'),
    title('Denoised Point Cloud - Median Filter Restoration (\sigma = 60)', 'FontSize', 26)
    
    
    
    
    
    
    
    
    %%
    
    
    
%     
%     
%     CoordinateMatrix=[];
% n = 1; 
% for jj =   1:col
%     for ii =   1:row
%         CoordinateMatrix = [CoordinateMatrix; [jj , test{12}(n), row-ii+1 ]];
%         n         = n+1;
%     end
% end
% disp('Buidling coordinates matrix finished.')
% disp('  ')
% 
% disp('Create the point cloud object using the function: pointCloud(.)')
% PointCloudObject = pointCloud(CoordinateMatrix,'Color',[zeros(row*col,2),ones(row*col,1)]);
% 
% 
% ptCloudOut1 = pcdenoise( PointCloudObject,'Threshold',1,'NumNeighbors',15);
% ptCloudOut2 = pcdenoise( PointCloudObject,'Threshold',0.5,'NumNeighbors',15);
% ptCloudOut3 = pcdenoise( PointCloudObject,'Threshold',0.1,'NumNeighbors',15);
%  
% 
% figure(6);
% subplot(2,2,1);pcshow([PointCloudObject ]);axis([1 col 300 500  1 row])
% title('Noisy PC')
% xlabel('Horizontal pixels');
% ylabel('Depth ')
% zlabel('Vertical pixels'); 
% subplot(2,2,2);pcshow([ptCloudOut1 ]);axis([1 col 300 500  1 row])
% title('PC denoise (1)')
% xlabel('Horizontal pixels');
% ylabel('Depth ')
% zlabel('Vertical pixels'); 
% 
% subplot(2,2,3);pcshow([ptCloudOut2 ]);axis([1 col 300 500  1 row])
% title('PC denoise (0.5)')
% xlabel('Horizontal pixels');
% ylabel('Depth ')
% zlabel('Vertical pixels'); 
% 
% subplot(2,2,4);pcshow([ptCloudOut3 ]);axis([1 col 300 500  1 row])
% title('PC denoise (0.1)')
% xlabel('Horizontal pixels');
% ylabel('Depth ')
% zlabel('Vertical pixels'); 
% 
