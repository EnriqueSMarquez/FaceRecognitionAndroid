function [faceImage] = ExtractFaceFromImage(image)

FDetect = vision.CascadeObjectDetector;
image = imresize(image,0.3);
image = rgb2gray(image);
%Read the input image
I = image;
[rowsImage,columnsImage] = size(I);
if columnsImage > rowsImage
    I = imrotate(I,90);
end
%Returns Bounding Box values based on number of objects
BB = step(FDetect,I);
sizeOfBiggerRec = 0;
for i = 1:size(BB,1)
   currentRectSize = (BB(i,3)*BB(i,4));
   if sizeOfBiggerRec < currentRectSize
        sizeOfBiggerRec = currentRectSize;
        currentRect = i;
   end
end
BB = BB(currentRect,:);
%%EXTRACT THE FACE
faceImage = I(BB(2):BB(2)+BB(4),BB(1):BB(1)+BB(3));

end