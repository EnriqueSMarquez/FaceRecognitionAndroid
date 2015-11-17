function [] = MatchPersonDatabase(database,image)

[~,dataSize] = size(database);
matches = zeros(1,dataSize);
for i=1:dataSize

    [~,matches(i)] = size(MatchSIFT(database{i},image));
    
end

[numberOfMatches,position] = max(matches);

display(strcat('ITS PERSON ', int2str(position),' WITH ', int2str(numberOfMatches),' MATCHES'))
end