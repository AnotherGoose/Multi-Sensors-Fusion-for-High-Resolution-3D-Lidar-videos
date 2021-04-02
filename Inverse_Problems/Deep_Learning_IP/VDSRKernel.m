close all
clc
clear all
load('trainedVDSR-Epoch-100-ScaleFactors-234.mat');

cd C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results
load Art_Depth_Reflectivity_Clean.mat DepthImage


% D_HR = imread("C:\Users\saifs\Downloads\RWMH-Instance.png")
D_HR = DepthImage
% D_HR = D_HR - 300
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
   
    
    test{loop} = Ivdsr{loop}
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
        %Fix some of these values and investigate results
        if(Ext >= 1)
            Ivdsr{loop}(i, j) = NaN;
%             denoisedI{loop}(i, j) = median(denoisedI{loop}(i-1:i+1,j-1:j+1), 'All','includenan');
        end

    
    end
end
% MedianTest{loop} = Ivdsr{loop}
% for i = 1:row
%     for j = 1:col
%     if any(isnan(Ivdsr{loop}(i, j)))
%         MedianTest{loop}(i, j) = nanmedian(Ivdsr{loop}(i-1:i+1,j-1:j+1), 'All');
%     end
%     end
% end


    x = kron(ones(row,1),(1:col)')
    y = reshape(Ivdsr{loop}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(2,3,loop)
    scatter3(x, y, z, 50 ,'.'),
    
    
    LRRescaled{loop} = double(imresize(Ilowres{loop}, [row col]))
    RMSE_VDSR(loop) = sqrt(mean(mean(((D_HR  - Ivdsr{loop}).^2),  'omitnan')));
    RMSE_LR(loop) = sqrt(mean(mean(((D_HR  - LRRescaled{loop}).^2))));
    RMSE_VDSRUnfilt(loop) = sqrt(mean(mean(((D_HR  - test{loop}).^2))));
    ssimval(loop) = ssim(Ivdsr{loop}, D_HR)
end



figure
plot(RMSE_LR, 'k')
hold on
plot(RMSE_VDSR, 'r--')
plot(RMSE_VDSRUnfilt, 'r')
xlabel('Index')
ylabel('RMSE')

figure
plot(ssimval)
xlabel('Index')
ylabel('SSIM')

imshow(Ivdsr{2}, [])

figure

    x = kron(ones(row,1),(1:col)')
    y = reshape(test{4}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,2,1)
    scatter3(x, y, z, 50 ,'.'),
    
    
    x = kron(ones(row,1),(1:col)')
    y = reshape(Ivdsr{4}(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(1,2,2)
    scatter3(x, y, z, 50 ,'.'),
%
