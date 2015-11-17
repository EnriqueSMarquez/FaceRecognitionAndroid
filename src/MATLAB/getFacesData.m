function [data] = getFacesData()

cd att_faces

for i=1:40

    nextFolder = strcat('s',int2str(i));
    cd(nextFolder)
    image = imread('1.pgm');
    cd ..
    imageV = ComputeEigenFacesVector(image);
    data(i,:) = imageV;
    
end

end