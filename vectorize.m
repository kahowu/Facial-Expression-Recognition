function feature_vector = vectorize (image)
%% Normalization
image = histeq(image);
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
features = cell2mat(features);
features = reshape(features, [40960, 1]);
feature_vector = features; 

end