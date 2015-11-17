function [database] = ReadDatabase()

cd SelfiesDataSet\

for i=1:8

    nextFolder = strcat('person',int2str(i));
    cd(nextFolder)
    database{i} = ExtractFaceFromImage(imread('1.jpg'));
    cd ..
end

end