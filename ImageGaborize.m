%% This function takes an image data set, gaborize it and save it as "GaborizedImages.mat"

function GaborizedImages = ImageGaborize(images)

num = size(images, 3);
GaborizedImages = zeros(40960, num);

fprintf('gaborizing...')

for i = 1 : num
    vec = vectorize(images(:,:,i));
    GaborizedImages(:, i) = vec;

end

fprintf('done.');
% save GaborizedImages GaborizedImages;