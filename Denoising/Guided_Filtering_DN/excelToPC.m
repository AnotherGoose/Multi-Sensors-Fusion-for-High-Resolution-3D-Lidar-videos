%required as calc. will throw errors
clear Ind

%needs to be changed back to outputBB.mat for system integration.
%For testing this can be replaced with the .mp4 provided.
Bounding = load('outputBB.mat');
%This is used for the Guide image.
vid = VideoReader('GuideImage.mp4'); 

%% User Input Varibles

%radius
r = input('Input Radius (5 is Recommended) ');

if isempty(r)
    r = 5
end

while r > 15
   Warning = 'Execution times will be above 30mins for this Radii'
    r = input('Please re-enter radius if you acknowledge the longer loading times: ')
    break
end

%regulization parameter
eps = input('Input Regularization parameter: ');
if isempty(eps)
    eps = 40
end
%% Frame looping

frames = vid.NumFrames;
fn = fieldnames(Bounding);


for FrameNum=1:numel(fn)
    %Loop through each frame
    if( isnumeric(Bounding.(fn{FrameNum})) )
        %% Adaptive Sampling converison for Guided filtering

        % Extract corresponding frame from perfect guide
        frame = im2gray(read(vid, FrameNum));
        frame = cast(frame,'double');

        [row, col] = size(frame);
        % Fast conversion to PC if needed
        vX = kron(ones(col,1),(1:row)');
        vY = kron((1:col)', ones(row,1));
        % Guide N x 3
        Ipc = [vY, -frame(:), -vX ]; 

           
        % Extract individual x, y and z coordinates
        xB = Bounding.(fn{FrameNum})(1,:);
        yB = Bounding.(fn{FrameNum})(2,:);
        zB = Bounding.(fn{FrameNum})(3,:);

        %Stub as AS input is not 100% polished
%         xB = round(xB);
%         yB = round(yB);
%         zB = round(zB);
        
        xB(xB < 0) = 0; 
        yB(yB < 0) = 0; 
        zB(zB < 0) = 0; 
        
        %To avoid NaN values
        xB(isnan(xB)) = 0;
        yB(isnan(yB)) = 0;
        zB(isnan(zB)) = 0;
        %resolution scale for the video input. If user isn't using the provided videos then resolution needs to be changed 
        xB(xB > 1024) = 1024;
        yB(yB > 436) = 436;
        zB(zB > 255) = 255; 
        
        v = [xB; imcomplement(zB); imcomplement(yB)];
        %Noisy Nx3
        ppc = v.';
        
        %% Guided filtering process
        
        %Number of points
        N = size(ppc,1);
        
        %Noisy point cloud
        NoisyPC(FrameNum) = pointCloud(ppc, 'Color', [zeros(N,2),ones(N,1)]);
        
        %Locating the neighbours for the indices matrix (Nxr^2)
        parfor n=1:N
            [Ind(n,:),dists] = findNearestNeighbors(NoisyPC(FrameNum),ppc(n,:),(2*r+1)^2-1);
        end
        %Guided filtering 
        Q = guidedfilterPC(Ipc,ppc,Ind,eps);
        
        %% Points clouds created for RMSE calculations
        
        PCoutput(k) = pointCloud(Q,'Color',[zeros(N,2),ones(N,1)]);
        %IPC(k) = pointCloud(Ipc,'Color',[zeros(N,2),ones(N,1)]);

    end
end

%% Plays all the denoised frames

%cannot use parfor loops
for f = 1:frames
    figure(1);
    pcshow(PCoutput(1,f))
    view(0,25)
    
end


%% Hidden within the comments are RMSE Calculations

%{
%% Used for RMSE calculations and plotting

% %ignore the tform and movingreg varibles 
% for f =1:size(fn,1)
%     %RMSE calc. for original Noisy input
%     [Noisy_tform,Noisy_movingReg,NoisyRMSE(f)] = pcregistericp(NoisyPC(1,f),IPC(1,f));
%     %RMSE calc. for Denoising output
%     [Out_tform,Out_movingReg,OutRMSE(f)] = pcregistericp(PCoutput(1,f),IPC(1,f));
% end
% 
% fig2 = figure(2);
% plot(NoisyRMSE); 
% grid on;
% grid  minor; 
% hold on; 
% plot(OutRMSE); 
% legend('\fontsize{12}Noisy','\fontsize{12}Processed'); 
% title('\fontsize{18}Processed vs Noisy RMSE (Radius = ' num2str(r) ' s)'); 
% xlabel('\fontsize{16}Frames');ylabel('\fontsize{16}RMSE');
%}
