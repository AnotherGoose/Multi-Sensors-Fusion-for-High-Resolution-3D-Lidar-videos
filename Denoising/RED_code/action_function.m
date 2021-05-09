%% Action Function
function [psnr_input,psnr_out,tEnd,ssim_out,abs_diff,output_im,RMSE_val,RMSE_noisy_val] = action_function(input_data,anaylsis_md,noise_level,reg_function,denoising_eng,deg_model,light_mode)

anaylsis_mode = anaylsis_md;

switch anaylsis_mode
    case 'image'        
        orig_im = double(imread(input_data));
        [output_im,deg_im,tEnd,psnr_out,ssim_out,abs_diff,psnr_input,RMSE_val,RMSE_noisy_val] = red_function(deg_model,...
                                                                            noise_level,...
                                                                            orig_im,...
                                                                            reg_function,...
                                                                            denoising_eng,...
                                                                            light_mode);
        % Display 3 Stage image in one plot 1.orig 2.deg 3.result
        figure();
        sgtitle("RED: " + reg_function + " " + denoising_eng);
        subplot(3,1,1);imshow(uint8(orig_im));title('Original Frame');
        subplot(3,1,2);imshow(uint8(deg_im));title("Degraded Frame, Noise Level: " + noise_level);
        subplot(3,1,3);imshow(uint8(output_im));title('Resulting Frame');                                                                 
    
    case 'frame'
        input_video = input_data;
        vid_in = VideoReader(input_video);
        
        i_frame = 50;
        frame = read(vid_in,i_frame);
        fprintf('Analysing Frame: %i\n',i_frame);
        orig_im = double(rgb2gray(frame));
        [output_im,deg_im,tEnd,psnr_out,ssim_out,abs_diff,psnr_input,RMSE_val,RMSE_noisy_val] = red_function(deg_model,...
                                                                            noise_level,...
                                                                            orig_im,...
                                                                            reg_function,...
                                                                            denoising_eng,...
                                                                            light_mode);  
        % Display 3 Stage image in one plot 1.orig 2.deg 3.result
        figure();
        sgtitle("RED: " + reg_function + " " + denoising_eng);
        subplot(3,1,1);imshow(uint8(orig_im));title('Original Frame');
        subplot(3,1,2);imshow(uint8(deg_im));title("Degraded Frame, Noise Level: " + noise_level);
        subplot(3,1,3);imshow(uint8(output_im));title('Resulting Frame');                                                                 
                                                                               
    case 'video'
        input_video = input_data;
        vid_in = VideoReader(input_video);
        blur_vid_out = VideoWriter('Blurred_vid/Blurred_out');
        result_vid_out = VideoWriter(['Result_vid/Result_vid_' reg_function denoising_eng]);
        
        FrameRate = 15;
        compT = 0;
        %...
        result_vid_out.FrameRate = 15;
        blur_vid_out.FrameRate = 15;
        open(blur_vid_out)
        open(result_vid_out)
        
        for i = 1:vid_in.NumFrames
            frame = read(vid_in, i);
            f = vid_in.NumFrames - i;
            fprintf('Analysing Frame: %i\n',i);
            fprintf('Frames Remaining: %i\n',f);            
            orig_im = double(rgb2gray(frame));
            [output_im,deg_im,tEnd,psnr_out,ssim_out,abs_diff,psnr_input,RMSE_val,RMSE_noisy_val] = red_function(deg_model,...
                                                                            noise_level,...
                                                                            orig_im,...
                                                                            reg_function,...
                                                                            denoising_eng,...
                                                                            light_mode);
            
            RMSE_val_str(i) = RMSE_val;
            RMSE_noisy_val(i) = RMSE_noisy_val;                                                            
            output_im = uint8(output_im);
            deg_im = uint8(deg_im);
            writeVideo(blur_vid_out, deg_im);
            writeVideo(result_vid_out, output_im);  
            compT = compT + tEnd;
            
        end
        close(blur_vid_out);
        close(result_vid_out);
        
    otherwise
        error('Anaylsis mode is not defined');
end




