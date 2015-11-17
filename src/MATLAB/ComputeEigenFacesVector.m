function [image1] = ComputeEigenFacesVector(image)

image1 = imresize(image,[50 50]);
image1 = image1';
image1 = image1(:);

end