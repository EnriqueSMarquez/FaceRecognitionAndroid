function [features, varargout] = extractVBOWFeatures(I,varargin)
%EXTRACTFVFEATURES Extract Visual Bag of Words features
% features = extractFVFeatures(I)

imsize = size(I);

% convert to grayscale facial image
if length(imsize)>2
    img = rgb2gray(I);
else
    img = I;
end


% single precision
img = single(img);
% extract sift features using vlfeat phow
[f, descrs]=vl_phow(img,'Step',1,'FloatDescriptors',true);

% take 50 random feature vectors
p= randperm(size(f,2));
f = f(:,p(1:1500));
descrs = descrs(:,p(1:1500));
% root SIFT
descrs_RootSIFT = sqrt(bsxfun(@rdivide,descrs,sum(descrs,2)));
% keypoints and descriptors
features.coords = [f(1,:)./imsize(1)-0.5;f(2,:)./imsize(2)-0.5];
features.descrs = descrs_RootSIFT;

end

