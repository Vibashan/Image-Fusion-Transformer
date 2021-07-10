
dir_source_ir  = dir("./images/ir/");
dir_source_vis = dir("./images/vis/");
dir_fused      = dir("./images/fuse/");

disp("Start");
disp('---------------------------Analysis---------------------------');

all = [0,0,0,0];
std_dev=0;
for i = 3:23;

    fused_image   = imread("./images/fuse/"+dir_fused(i).name);
    source_image1 = imread("./images/ir/"+dir_source_ir(i).name);
    source_image2 = imread("./images/vis/"+dir_source_vis(i).name);
    
    [EN,MI,SCD,MS_SSIM] = analysis_Reference(fused_image,source_image1,source_image2);
    all = all + [EN,MI,SCD,MS_SSIM];
end

disp(all/21);
disp('Done');