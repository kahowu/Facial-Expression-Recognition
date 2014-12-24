function feature_vector = vectorize (image)
%% Normalization
% visualize(image);
% image = histeq(image);
% visualize(image);
% image = double(image);
% max_p = max(max(image));
% min_p = min(min(image));
% image = uint8((image - min_p) * 255 /(max_p - min_p));
%%


load gabor_filters;
features = cell(5,8);

% Apply Gabor filters 
for s = 1:5
    for j = 1:8
        input = abs(ifft2(G{s,j}.*fft2(double(image)),32,32));
        max_p = max(max(input));
        min_p = min(min(input));
        output = ((input-min_p)/(max_p-min_p) - 0.5) * 2;
        features{s,j} = output;
    end
end    

% Dimensionality reduction
features_reduced  = cell2mat(features);
features_reduced (5:5:end,:) = [];
features_reduced (3:3:end,:) = [];
features_reduced (:,5:5:end) = [];
features_reduced (:,3:3:end) = [];
feature_vector = reshape(features_reduced, [86*137, 1]);

end