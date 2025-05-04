% Creates an image of a 2-D Gaussian 
% arguments:  D - Number of pixels of the side of the square image
%                           Sigma -  Standard deviation of the Gaussian (in number of pixels)
%                           C - A 2-D vector specifying the coordinates of
%                                the center of the Gaussian (in pixels), with
%                                respect to the center of the image

function M = GausSpot(D,Sigma,C)

X = ones(D,1) * (1:D) - (D+1)/2;
Y = flipud((1:D)') * ones(1,D) - (D+1)/2;

dist = sqrt((X - C(1)).^2 + (Y - C(2)).^2);

M = normpdf(dist,0,Sigma);
