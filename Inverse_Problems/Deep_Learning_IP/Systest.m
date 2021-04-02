4
clear all
close all
clc
tic

data = 0;
data = input('Mann data (1), Monkey data (2), Custom data (3):\n  ', 's');
ver = 0;
while(ver == 0)
if(data=='1')
    cd C:\Users\saifs\Desktop\Backup\UNI\LIDAR\Shared_4thYear_MEng_Projects\Shared_4thYear_MEng_Projects\Tutorial_1_LidarRawData\results
    load Face_Depth_Reflectivity_Clean.mat DepthImage
    D_HR = DepthImage
    [row,col] = size(D_HR);
    scatter3(kron(ones(row,1),(1:col)'),reshape(D_HR(end:-1:1,:)',row*col,1)',kron((1:row)',ones(col,1)),50 ,'.'),
    ver = 1
%     D_HR = D_HR - min(D_HR(:))
    caxis([300 500])
    elseif(data=='2')
        DepthImage = rgb2gray(imread('C:\Users\saifs\Desktop\LIDARTesting\MonkeyDepth100.png'))
        D_HR = DepthImage
        [row,col] = size(D_HR);
        ver = 1
        subplot(2,2,1)
        scatter3(kron(ones(row,1),(1:col)'),reshape(D_HR(end:-1:1,:)',row*col,1)',kron((1:row)',ones(col,1)),50 ,'.'),
        ref3d = D_HR
    elseif(data=='3')
    DepthImage = rgb2gray(imread('C:\Users\saifs\Desktop\LIDARTesting\AudiR8300T.png'))
    D_HR = DepthImage
    ver = 1
    
    [row,col] = size(D_HR);
    reconst = zeros(row, col)
    parfor i = 1:1:row
    for j = 1:1:col
    nat = impixel(D_HR, i, j)
    reconst(i, j) = nat(1)      %Investigate double(D_HR)
    
    end
    end
    
    reconst = reconst'
    x = kron(ones(row,1),(1:col)')
    y = reshape(reconst(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    ref3d = reconst
subplot(2,2,1)
scatter3(x, y, z, 50 ,'.'),
    
else
    disp('NO FILE')
    ver = 0;
    return
end
end


op = '';
op = input('What operation do you want to perform? Denoising/Super-Resolution\n', 's')
if(op=='DN')
denoiseNet = denoisingNetwork('DnCNN')
noisyI = imnoise(D_HR, 'gaussian', 0, 0.01)

    [row,col] = size(D_HR);
    reconst = zeros(row, col)
    parfor i = 1:1:row
    for j = 1:1:col
    nat = impixel(D_HR, i, j)
    reconst(i, j) = nat(1)
    
    end
    end

x = kron(ones(row,1),(1:col)')
y = reshape(noisyI(end:-1:1,:)',row*col,1)'
z = kron((1:row)',ones(col,1))

subplot(2,2,2)
scatter3(x, y, z, 50 ,'.'),

denoisedI = denoiseImage(noisyI,denoiseNet);


    [row,col] = size(D_HR);
    reconst = zeros(row, col)
    parfor i = 1:1:row
    for j = 1:1:col
    nat = impixel(D_HR, i, j)
    reconst(i, j) = nat(1)
    
    end
    end

x = kron(ones(row,1),(1:col)')
y = reshape(denoisedI(end:-1:1,:)',row*col,1)'
z = kron((1:row)',ones(col,1))


subplot(2,2,3)
scatter3(x, y, z, 50 ,'.'),
figure
title('Reference Image vs Noisy Image vs Denoised Image')
montage({D_HR, noisyI, denoisedI}, 'Size',[1,3])

clear DepthImage
[row,col] = size(D_HR);

reconst = reconst'
   CoordinateMatrix=[];
n = 1; 
for jj =   1:col
    for ii =   1:row
        CoordinateMatrix = [CoordinateMatrix; [jj , double(denoisedI(n)), row-ii+1 ]];
        n         = n+1;
    end
end

PointCloudObject = pointCloud(CoordinateMatrix,'Color',[zeros(row*col,2),ones(row*col,1)]);

ptCloudOut1 = pcdenoise( PointCloudObject,'Threshold',0.3,'NumNeighbors',15);
figure
pcshow(ptCloudOut1)

toc


elseif(op=='SR')
        
load('trainedVDSR-Epoch-100-ScaleFactors-234.mat');
Ireference = D_HR

scaleFactor = 0.25;
Ilowres = imresize(Ireference,scaleFactor,'bicubic');
IlowresC = cat(3, Ilowres, Ilowres, Ilowres)
Iycbcr = rgb2ycbcr(IlowresC);
Iy = Iycbcr(:,:,1);
Icb = Iycbcr(:,:,2);
Icr = Iycbcr(:,:,3);

[nrows,ncols,np] = size(Ireference);
Ibicubic = imresize(Ilowres,[nrows ncols],'bicubic');

Iy_bicubic = double(imresize(Iy,[nrows ncols],'bicubic'));
Icb_bicubic = imresize(Icb,[nrows ncols],'bicubic');
Icr_bicubic = imresize(Icr,[nrows ncols],'bicubic');

Iresidual = activations(net,Iy_bicubic,41);
Iresidual = double(Iresidual);

Isr = Iy_bicubic + Iresidual;

Ivdsr = ycbcr2rgb(cat(3,Isr,Icb_bicubic,Icr_bicubic));
figure
montage({Ireference, Ilowres, Ivdsr}, 'Size',[1,3])
title('Reference Image | Low-Res Component | VDSR Upscaled Product')

%   reconst = reconst'
    x = kron(ones(row,1),(1:col)')
    y = reshape(ref3d(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))
    figure
    subplot(2,2,1)
    scatter3(x, y, z, 50 ,'.'),


    [row,col] = size(rgb2gray(IlowresC));
    reconst = zeros(row, col)
    parfor i = 1:1:row
    for j = 1:1:col
    nat = impixel(IlowresC, i, j)
    reconst(i, j) = nat(1)
    
    end
    end
    reconst = reconst'
    x = kron(ones(row,1),(1:col)')
    y = reshape(reconst(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))
    
    subplot(2,2,2)
    scatter3(x, y, z, 50 ,'.'),
    clear reconst
    
    [row,col] = size(rgb2gray(Ivdsr));
    reconst = zeros(row, col)
    parfor i = 1:1:row
    for j = 1:1:col
    nat = impixel(Ivdsr, i, j)
    reconst(i, j) = nat(1)
    
    end
    end
    reconst = reconst'
    x = kron(ones(row,1),(1:col)')
    y = reshape(reconst(end:-1:1,:)',row*col,1)'
    z = kron((1:row)',ones(col,1))

    subplot(2,2,3)
    scatter3(x, y, z, 50 ,'.'),
    
    CoordinateMatrix=[];
    
n = 1; 
for jj =   1:col
    for ii =   1:row
        CoordinateMatrix = [CoordinateMatrix; [jj , double(Ivdsr(n)), row-ii+1 ]];
        n         = n+1;
    end
end

PointCloudObject = pointCloud(CoordinateMatrix,'Color',[zeros(row*col,2),ones(row*col,1)]);

ptCloudOut1 = pcdenoise( PointCloudObject,'Threshold',0.3,'NumNeighbors',15);
figure
pcshow(ptCloudOut1)
end
toc


