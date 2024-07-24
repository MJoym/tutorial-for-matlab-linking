% function choose_best_trans_left(all_anchor_points)
clear all
% added a change
% This script is to be used after performing the optimization.
% This script shows visually the transformations according to the
% optimization performed (minimal RMSDs calculated in
% optimization_joanna.mat).
% This script can be used as a function (just uncomment first and last rows)
% Instructions for use:
% 1) If you are using this script after optimization then uncomment the
% section "My parameters (optimization points chosen)"
% Load retino matrix (make sure it is in your path).
% Change "opt_date" to your actual date (if used "optimization_joanna.m" or
% GUI by Joanna - If not change accordingly to your needs).
% "path_winners" should be your path to the winner parameters file.
% Load the optimization points you chose/have.
% "mat_row" should be changed if you want to visualize a specific result from all the winners.
% Make sure your anchor & opt. points are yours and/or correct.
%
% 2) If you want to choose manually your parameters then use section "Choose
% Points Manually". Uncomment this section and comment
% "My parameters (optimization points chosen)" and "Amit's parameters" 
% Change "anchor_points", "winner" and "min_rmsd" manually.
%
% 3) If you want to use Amit's parameters uncomment section called
% "Amit's parameters" and comment the two other sections mentioned above.
% 
% 4) For "Create Stimulus" section change the condition ("cond") and stimulus
% type ("stimType") you want to visualize its transformation.
% Read the instructions in this section for "stimType".
% If you are using stimuli other than letters ('G', 'A', 'N', 'D', 'L', 'F')
% and cartesian/polar grids you will have to add other letters/write your
% own "create stimulus" script.
% If you need to visualize a vertical line, comment the part for letters
% creation and uncomment %%% Vertical Bar %%% section.
%
% 5) "Upload .png files for comparison" section:
% Upload relevant images for you (chamber/VSDI/Intrinsic maps).
%
% 6) "Perform Transformation" section:
% After you change all the paramters explained above there is no need to
% change anything in this section. Run the script.
% Optional: If you want to visualize diferently than the output explained
% below then change the subplots accordingly.
%
% OUTPUT: Three images in one figure (Stimulus on visual field,
% transformation shown on chamber and transformation shown on VSDI image).

% Written by Joanna M. F.
% Last update: 10/6/22 - Joanna Molad F.

%% Basic parameters (See that these parameters match your model):
% Setting global parameters:
monkey = 'Gandalf';

% Set Stimulus Parameters:
% 1) "cartGrid" - Draw cartesian coordinates grid for stimulus
% 2) "polarGrid" - Draw polar coordinates grid for stimulus
% 3) 'G', 'A', 'N', 'D', 'L', 'F' - Draw letter for stimulus
% 4) Choose sesion (Either 'a' or 'd')
% 5) For vertical bar: stimType = 'VerticalBar';

stimType = 'L';
session = 'd'; 
cond = 6;
b = 201;
model = 2;
reg = 0; % If another registration is wanted (this performs a linear transformation -> most of the times: linear is not correct)
NAN_PIXEL_VAL = -5;

if monkey == 'Boromir'
    load('opt_points_241121_7_8_22.mat');
    load('retino_mat_Boromir_August.mat');
    opt_date = '15-08-2022-18';
    date = '241121';
    n = 540;
    m = 960;
    stim = zeros(n,m);
    path_winners = ['C:/Users/joann/Documents/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/Opt_results/',opt_date,'/'];
    VSDI_IM = imread(['/media/joy/My Passport/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/images/',date,'/Session ',session,'/',stimType,'.png']);
%     chamber = imread(['/media/joy/My Passport/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/images/',date,'/',date,'_BV_resized.png']);
    chamber = imread(['D:/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/images/',date,'/',date,'_BV_resized.png']);
%     VSDI_IM = imread(['D:/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/VSDI points/',date,'/Session ',session,'/cond',num2str(cond),'_resized.png']);
    load([path_winners,'min_50_rmsd_winners.mat']); % Load the winners from the optimization script.
    leftH = false;
    quickM = false;
    lowH = true;
    ppd = 50; % Pixels per degree (Gandalf = 35, Boromir = 50)
    [VMempirical, empiricalVM_area, VM_VF] = settingVM(32, leftH, stim);
end

if monkey == 'Gandalf'
    load('opt_points_8_5_22.mat')
    load('retino_mat_20_03_22.mat');
    opt_date = '25-07-2022-17';
    date = '201118';
    n = 301;
    m = 512;
    stim = zeros(n,m);
    path_winners = ['/media/joy/My Passport/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/Opt_results/', date,'/', opt_date,'/min_50_rmsd_winners.mat'];
%     path_winners = ['C:/Users/joann/Documents/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/Opt_results/', date,'/', opt_date, '/min_50_rmsd_winners.mat'];
    VSDI_IM = imread(['/media/joy/My Passport/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/images/',date,'/Session ',session,'/',stimType,'.png']);
%     VSDI_IM = imread(['D:/Neuroscience Master/Retinotopic Model\Retinotopic model Joanna/',monkey,'/images/',date,'/Session ',session,'/',stimType,'.png']);
% %     VSDI_IM = imread(['D:/Figures for CrelDraw/After Corel/Trans_',stimType,'_cond',session,'_corel_resized','.jpg']);
%     VSDI_IM = imread(['C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\Gandalf\images\201118\cond ',session, '\',stimType,'_new_resized.png']);
    chamber = imread(['/media/joy/My Passport/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/images/',date,'/',date,'_BV_resized.png']);
%     chamber = imread(['C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\',monkey,'\Code\Amit old code\2011-BV.png']);
    load(path_winners); % Load the winners from the optimization script.
    leftH = true;
    quickM = false;
    lowH = true;
    ppd = 35; % Pixels per degree (Gandalf = 35, Boromir = 50)
    [VMempirical, empiricalVM_area, VM_VF] = settingVM(22, leftH, stim);
end

%% My parameters (optimization points chosen):

% min_rmsd_winners = min_winners_VMarea;
mat_row = 1;
% Anchor points according to file saved:
anchor_points_temp = [min_rmsd_winners(mat_row,1), min_rmsd_winners(mat_row,2); min_rmsd_winners(mat_row,3), min_rmsd_winners(mat_row,4)];

% Find the brain coordinates regarding the anchor_points_temp (Get cartesian brain coords):
if opt_points(1,1) == 0 && opt_points(1,2) == 0 && opt_points(2,1) == 0 && opt_points(2,2) == 0
    cart_points = conv_pix_deg(ppd, [opt_points(:,3), opt_points(:,4)], 0);
    cart_pointsx = check_coords(cart_points(:,2)', 'x', stim, 0);
    cart_pointsy = check_coords(cart_points(:,1)', 'y', stim, 0);
    opt_points(:,1) = cart_pointsy;
    opt_points(:,2) = cart_pointsx;
%     anchor_points_temp(:,2) = 935;
    r = find(opt_points(:,1) == anchor_points_temp(1,1) & opt_points(:,2) == anchor_points_temp(1,2));
    r2 = find(opt_points(:,1) == anchor_points_temp(2,1) & opt_points(:,2) == anchor_points_temp(2,2)); 
else
    r = find(opt_points(:,1) == anchor_points_temp(1,1) & opt_points(:,2) == anchor_points_temp(1,2));
    r2 = find(opt_points(:,1) == anchor_points_temp(2,1) & opt_points(:,2) == anchor_points_temp(2,2)); 
end
% Complete anchor points both in VF and brain coords:
anchor_points = [min_rmsd_winners(mat_row,1), min_rmsd_winners(mat_row,2), opt_points(r,5), opt_points(r,6); min_rmsd_winners(mat_row,3), min_rmsd_winners(mat_row,4), opt_points(r2,5), opt_points(r2,6)];
% anchor_points(:,2) = 935;
winner = [min_rmsd_winners(mat_row,5), min_rmsd_winners(mat_row,6), min_rmsd_winners(mat_row,7)];
min_rmsd = min_rmsd_winners(mat_row,8);
% Get polar VF reg. coordinates:
% anchor_pointsx = check_coords(anchor_points_temp(:,2)', 'x', stim, 1);
% anchor_points2 = anchor_points;
% anchor_points2(:,2) = anchor_pointsx';
% anchor_pts_deg = conv_pix_deg(ppd,anchor_points2, 1);
anchor_pts_deg = conv_pix_deg(ppd,anchor_points, 1);
% points_tempx = check_coords(points_temp(:,1)', 'x', stimulus, 0);
% points_tempy = check_coords(points_temp(:,2)', 'y', stimulus, 0);

ys = [anchor_pts_deg(1,1), anchor_pts_deg(2,1)];
xs = [anchor_pts_deg(1,2), anchor_pts_deg(2,2)];
%xs = [1, 1];
if lowH
    ys = -(abs(ys));
end
if ~leftH
    xs = -(abs(xs));
end

% Separate to new params the cartesian brain points:
ybrain_row = [anchor_points(1,3), anchor_points(2,3)];
xbrain_col = [anchor_points(1,4), anchor_points(2,4)];
%% Choose Points Manually:
% 
% anchor_points = [25, 32, 41, 48; 42, 66, 51, 80];
% winner = [0.06, 0.5, 1];
% min_rmsd = 4.8442;
% Get polar VF reg. coordinates:
% anchor_pts_deg = conv_pix_deg(35,anchor_points, 1);
% ys = [anchor_pts_deg(1,1), anchor_pts_deg(2,1)];
% xs = [anchor_pts_deg(1,2), anchor_pts_deg(2,2)];
% ys = -(abs(ys));
% 
% Get cartesian brain points:
% ybrain_row = [anchor_points(1,3), anchor_points(2,3)];
% xbrain_col = [anchor_points(1,4), anchor_points(2,4)];

%% Amit's parameters:
% Picking 2 best points for registration
% point1=1;
% point2=2;
% load('winner201118.mat')
% alpha = s;
% % Joy: Two anchor points already calculated for the registration:
% xbrain_col = [ points(point1, 1) points(point2, 1)]; % x coord on the brain
% ybrain_row = [ points(point1, 2) points(point2, 2)]; % y coord on the brain
% xs = [ points(point1, 3) points(point2, 3)]; % x coord on the stimulus
% ys = [ points(point1, 4) points(point2, 4)]; % y coord on the stimulus
% winner = [a, alpha, k];
% min_rmsd = 0.71;
% % Make sure ys are negative - it's always lower hemifield
% ys = -(abs(ys));

%% Stimulus Creation:
stimType = 'VerticalBar';

if strcmpi(stimType, 'dot')
    dot_size = 0.5;
%     square_size = 1;
    dot_center = [-0.5 -1.5]; % [X,Y] coords
    stim = create_dot(dot_size, dot_center, n, m);
elseif strcmpi(stimType, 'VM')
    stim(VM_VF(:,1), VM_VF(:,2)) = 255;
    stim(stim==0)=130/255;
elseif strcmpi(stimType, 'VerticalBar')
    vl_center = [0.9, -0.7]; % for session a: vl_center = [0.6, -0.75] (for session d the center is vl_center = [1.4, -0.7])
    vl_size = 1; % In degrees
    stim = vertical_line(vl_center, vl_size, ppd); % for vertical line the center is [0.9,-0.7]
else
    stim = create_stimulus(stimType, session, ppd, leftH);% Go to "create_letter_stim.m" to change your path to .bmp files needed for this.

end
% load('vertical_bar_conv_psf.mat');
figure; imshow(stim); title('Original Stimulus on VF')
%% PSF
% load('D:\Neuroscience Master\Encoding\Joy\Gandalf\PSF\201118\PSF_not_normalized.mat');
load('/media/joy/My Passport/Neuroscience Master/Encoding/Joy/Gandalf/PSF/201118/PSF_not_normalized.mat');
% figure; surf(psf_cropped); colormap;

% norm_psf = psf_cropped./sum(psf_cropped(:));
% figure; surf(norm_psf); colormap;
% fs = 0.001;
% norm_psf = highpass(norm_psf,fs);
norm_psf = psf_cropped;
norm_psf = mfilt2(norm_psf, 36, 1, 1.2);
% figure; surf(norm_psf); colormap;
norm_psf(norm_psf < 0) = 0;
% figure; surf(norm_psf); colormap;
norm_psf = norm_psf - min(min(norm_psf));
norm_psf = norm_psf./max(max(norm_psf));
norm_psf = norm_psf(5:33,5:33);
% figure; surf(norm_psf); colormap;
norm_psf(norm_psf < 0.368) = 0;
norm_psf = norm_psf - min(min(norm_psf));

% figure; surf(norm_psf); colormap;

%% Convolution with PSF
paddingSize = floor(size(norm_psf, 1)/2);
bgValue = stim(1,1);
stim_padded = ones(size(stim)+2*paddingSize)*bgValue;
stim_padded(paddingSize+1:paddingSize+size(stim,1), paddingSize+1:paddingSize+size(stim,2)) = stim;
% figure; imagesc(stim_padded);axis image; axis off; colormap gray; title('stimulus padded');
                
convI = conv2(stim_padded, norm_psf, 'valid');
% load('vertical_bar_conv_psf.mat');
stimulus = convI;
stimulus = stimulus(1:end-1, 1:end-1);
stimulus = stimulus/max(max(stimulus));
% for i = 1:size(stimulus,1)
%     for j = 1:size(stimulus,2)
%         if stimulus(i,j) > stimulus(1,1)
%             stimulus(i,j) = stimulus(i,j)*255;
%         end
%     end
% end
% % stimulus(stimulus > stimulus(1,1)) = stimulus*255;
% stimulus(stimulus < 0.96) = 0;
% stimulus = stimulus*255;
% stimulus = stimulus/min(min(stimulus));
% stim_conv = stimulus; 

figure; imshow(stimulus(1:100,1:100)); title('Stim after convolution with PSF'); colormap(mapgeog);
% figure(); imagesc(stim_conv);axis image; axis off; title('stimulus blurred with PSF'); colormap(mapgeog)

%% inv-sigmoid non-linearity (Zurawel et al. 2016)
% madRange = 0.5:0.25:10;
% madRange = 4;
% numMADs = madRange;
% % for numMADs = madRange
% splotNum= 3;
% valInd = find(numMADs == madRange); %which iteration are we in
% validPixels = [bgValue; stim_conv(roundn(stim_conv, -3)~=bgValue)];
% ISRange =  [median(validPixels) - numMADs*mad(validPixels) median(validPixels) + numMADs*mad(validPixels)]; % resitant to noise
% % ISRange =  minmax(stim_conv(roundn(stim_conv, -3)~=bgValue)'); %doesn't consider noise in the stimulus
% [stim_conv_nl rangeM] = invSig(stim_conv,1,0,ISRange);%2.193e+4, 0, ISRange);
% % [stim_conv_nl rangeM] = invSig(stim_conv,1, 0, ISRange, numSTDs); %option of "clipping" output to +/- x stds
% figure(); imagesc(stim_conv_nl);axis image; axis off; colormap gray; title('blurred stimulus after non-linearity');
% cxVals = caxis; %this is arbitrary... test before presentation/figure
% cxVals(2) = 1.1*abs(cxVals(1));
% caxis(cxVals);
%% Collinearity Index (CI):
% Give a weight for each pixel in the VF image 
% according to its collinearity with respect to its neighbors.
weights_vec = CI_weights(stim);


%% Non-classical RF
% Look for the weights given for each pixel in the VF inside the cRF with
% respect to the ncRF and give a suppression index:
collinear_suppression_map = collinearSuppressionWeights(weights_vec);
col = zeros(pixelX, pixelY); % preallocating for speed
for pixelX = 1:size(collinear_suppression_map,1)
    for pixelY = 1:size(collinear_suppression_map,2)
        col(pixelX, pixelY) = Pl + median(collinear_suppression_map(pixelX,pixelY) .* Y(pixelX,pixelY));

    end
end
%% Naka-Rushton non-linearity
q = 2; % according to past literature
colNR = (stimulus).^q ./ ((stimulus).^q+(col).^q);


%% Upload .png files for comparison:
% VSDI_IM = imread('C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\pre processed data\gandals_161018a\gandalf_chamber_cond4.png');
% VSDI_IM = imread(['C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\',monkey,'\VSDI points\',date,'\session ',session,'\cond',num2str(cond),'_resized.png']);
% chamber = imread(['C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\',monkey,'\VSDI points\130422a\130422a_BV_tranformed_to_241121_resized.png']);
% VSDI_IM = imread(['C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\',monkey,'\VSDI points\130422a\cond 4\square_yarden_reg_resized.png']);
chamber = imsharpen(chamber, 'Amount', 2);
if ~exist('stimulus', 'var')
    stimulus = stim;
end
%% Perform Transformation: 

for i = 1:size(winner,1)
    aVal = winner(i,1);
    alphaVal = winner(i,2);
    kVal = winner(i,3);
    [out_img, OM, xcoord, ycoord, tform] = trans_and_registration(stimulus,model,aVal,b,alphaVal,kVal,retino_mat,xbrain_col,ybrain_row,xs,ys,leftH,lowH,quickM, ppd);
    trans_img = out_img(:,:,1);
    trans_img(trans_img==0) = nan;
    trans_img(isnan(trans_img)) = 0;
%     figure; imshow(trans_img, [0.2 1]); title("Transformed Predicted Vertical Bar on Cortex"); colormap(mapgeog);
%     trans_img = imsharpen(trans_img, 'Amount', 2);

%     save_path_winners = ['C:/Users/joann/Documents/Neuroscience Master/Retinotopic Model/Retinotopic model Joanna/',monkey,'/Opt_results/', opt_date,'/'];
%     cd(save_path_winners);
%     save(['tform_',opt_date,'.mat'],'tform');

%     rmsd_area = calc_VM_rmsd(xcoord, ycoord, tform, empiricalVM_area, VM_VF);
%% 
% blue_chamber = imread('C:\Users\joann\Documents\Neuroscience Master\Retinotopic Model\Retinotopic model Joanna\Gandalf\images\201118\frame33_blank_resized.png');
%     figure;imshowpair(trans_img,VSDI_IM, 'blend'); axis image; title(['stim ', stimType, ' after trans. on chamber','. Session', session]);

    figure(); 
%     subplot(1,3,1); imshow(stimulus(1:100,1:100)); axis image;title('stimulus');
    subplot(1,2,1);imshow(trans_img); axis image; title(['stim ', stimType, ' after trans. on chamber']); colormap(mapgeog);
    figure(); 
%     subplot(1,3,1); imshow(stimulus(1:100,1:100)); axis image;title('stimulus');
    subplot(1,2,1);imshow(trans_img); axis image; title(['stim ', stimType, ' after trans. on chamber']); colormap(mapgeog);
%     subplot(1,2,2);imshow(VSDI_IM); axis image; title(['stim ', stimType, ' empirical']);subplot(1,2,2);imshow(VSDI_IM); axis image; title(['stim ', stimType, ' empirical']);

%     subplot(1,3,2);
    subplot(1,2,2); imshowpair(trans_img,VSDI_IM,'blend'); axis image; title(['stim ', stimType, ' after trans. on cortex', '. Session', session]); colormap(mapgeog);
%     b = axes;
%     %// Set the title and get the handle to it
%     ht = title(['Joy, Session ',session,', cond #',num2str(cond),': a=',num2str(aVal),' K=',num2str(kVal),' alpha=',num2str(alphaVal), ' RMSD=',num2str(min_rmsd)]);
%     %// Turn the visibility of the axes off
%     b.Visible = 'off';
%     %// Turn the visibility of the title on
%     ht.Visible = 'on';
%     
% %     figure; imshowpair(trans_img,chamber,'blend'); axis image;
% 
end