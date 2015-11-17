 clear all
clc
%Detect objects using Viola-Jones Algorithm

%To detect Face
FDetect = vision.CascadeObjectDetector;
image = imread('2.jpg');
image = imresize(image,0.3);
image = rgb2gray(image);
%Read the input image
I = image;

%Returns Bounding Box values based on number of objects
BB = step(FDetect,I);
figure,
imshow(I); hold on
sizeOfBiggerRec = 0;
for i = 1:size(BB,1)
   currentRectSize = (BB(i,3)*BB(i,4));
   if sizeOfBiggerRec < currentRectSize
        sizeOfBiggerRec = currentRectSize;
        currentRect = i;
   end
end
BB = BB(currentRect,:);
 rectangle('Position',BB,'LineWidth',5,'LineStyle','-','EdgeColor','r');
title('Face Detection');
hold off;
%%EXTRACT THE FACE
faceImage = I(BB(2):BB(2)+BB(4),BB(1):BB(1)+BB(3));
figure
imshow(faceImage)
[siftFeatures,siftDescriptors] = vl_sift(single(faceImage));

