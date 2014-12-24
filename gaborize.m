%% NOTE: Generates gaborized images of labeled_images, public_test_images, 
%% and hidden_test_images for "solution.py"
load public_test_images;
load labeled_images;
load hidden_test_images;
TrainImages = ImageGaborize(tr_images);
save TrainGaborized TrainImages;
PublicImages = ImageGaborize(public_test_images);
save PublicGaborized PublicImages;
HiddenImages = ImageGaborize(hidden_test_images); 
save HiddenGaborized HiddenImages;


%% To gaborize additional dataset:
%% 1) Load the mat file
%% 2) Input image data into ImageGaborize function 
%% 3) Save the gaborized images as the name you desired