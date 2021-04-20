clear all
close all
tic

addpath C:\Users\saifs\Downloads
addpath C:\Users\saifs\Downloads\noisy_videos

v = VideoReader('newNDLR5.mp4.avi');
Reference = VideoReader('DLR.mp4');
% Bounding = read(v);
% % imshow(Bounding);
% Bounding = rgb2gray(Bounding)
% [row, col] = size(Bounding);
% Bounding = imcomplement(Bounding);
% delMat = ones(row, col);
load net4
% denoisedI = denoiseImage(Bounding, net4);
% 
% denoisedI = double(denoisedI);

Bounding = read(v, 1);
Bounding = rgb2gray(Bounding)
[row, col] = size(Bounding);

parfor frame = 1:v.NumFrames
    Bounding = read(v, frame);
    Bounding = rgb2gray(Bounding)
    Bounding = imcomplement(Bounding)
    
    Ref = read(Reference, frame);
    Ref = rgb2gray(Ref)
%     Ref = cast(Ref,'double');
%     Ref = Reference;
    Ref = imcomplement(Ref);
    
    [row, col] = size(Bounding);
    denoisedI{frame} = denoiseImage(Bounding, net4);
    denoisedI{frame} = double(denoisedI{frame})
    origFrame{frame} = denoisedI{frame}
    delMat = ones(row, col);
 for i = 2:1:row - 1
    for j = 2:1:col - 1
       Ext = 0;
       North = denoisedI{frame}(i-1, j);
       West = denoisedI{frame}(i, j-1);
       East = denoisedI{frame}(i, j+1);
       South = denoisedI{frame}(i+1, j);
       
       diffN = abs(denoisedI{frame}(i,j) - North);
       diffW = abs(denoisedI{frame}(i,j) - West);
       diffE = abs(denoisedI{frame}(i,j) - East);
       diffS = abs(denoisedI{frame}(i,j) - South);
       
        if(diffN >= 3)
           Ext = Ext + 1;
        end
        if(diffN >= 3)
           Ext = Ext + 1;
        end
        if(diffN >= 3)
           Ext = Ext + 1;
        end
        if(diffN >= 3)
           Ext = Ext + 1;
        end
        
        if(Ext >= 1)
        delMat(i, j) = NaN;
        end
        
    
    end
 end
 denoisedI{frame} = double(denoisedI{frame})
denoisedI{frame} = denoisedI{frame}.*delMat;
 
Bounding = double(Bounding)
Ref = double(Ref)

% 
MedianTest{frame} = denoisedI{frame};
for i = 2:row-1
    for j = 2:col-1
    if any(isnan(denoisedI{frame}(i, j)))
        MedianTest{frame}(i, j) = nanmedian(denoisedI{frame}(i-1:i+1,j-1:j+1), 'All');
    end
    end
end


RMSE_Processed(frame) = sqrt(mean(mean((MedianTest{frame}  - Ref).^2), 'omitnan'));  
RMSE_Noisy(frame) =  sqrt(mean(mean((Bounding  - Ref).^2)));  




end
toc
pause
% 
% figure
% montage({uint8(Bounding), uint8(denoisedI{frame})})

figure
    x = kron(ones(row,1),(1:col)');
    y = reshape(Bounding(end:-1:1,:)',row*col,1)';
    z = kron((1:row)',ones(col,1));
 
    subplot(2,1,1)
    scatter3(x, y, z, 1 ,y),;
    
    
    x = kron(ones(row,1),(1:col)');
    y = reshape(MedianTest{1}(end:-1:1,:)',row*col,1)';
    z = kron((1:row)',ones(col,1));

    subplot(2,1,2)
    scatter3(x, y, z, 1 ,y),;

    
%%    
figure
for i = 1:v.numFrames
Bounding = rgb2gray(read(v, i))
x = kron(ones(row,1),(1:col)');
    y = reshape(Bounding(end:-1:1,:)',row*col,1)';
    z = kron((1:row)',ones(col,1));
    scatter3(x, y, z, 1, y),;
% colormap(jet(length(y)))
% colorbar;
view(0, 25)
pause(0.033)
end



for i = 1:v.numFrames
x = kron(ones(row,1),(1:col)');
    y = reshape(MedianTest{i}(end:-1:1,:)',row*col,1)';
    z = kron((1:row)',ones(col,1));
    scatter3(x, y, z, 1, y),;
% colormap(jet(length(y)))
% colorbar;
view(0, 25)
pause(0.033)
end


figure
plot(RMSE_Noisy, 'k')
hold on
plot(RMSE_Processed, 'r')
grid
grid minor
title('High Noise - RMSE')
legend('Noisy', 'Processed')
xlabel('Frame')
ylabel('RMSE')

    