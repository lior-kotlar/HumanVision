
% Gaussian spot
im1=GausSpot(64,5,[0 0]);
    imagesc(im1);
    im2=GausSpot(64,5,[2 1]);
    imagesc(im2);
    v=LK_alg(im1,im2, 0, ones(64), [0 0]', 5) 

   % flower garden 
    i1 = double(imread('flower-i1.tif'));
  i2 = double(imread('flower-i2.tif')); 
  
  mask = zeros(120, 180); 
  mask(20:40, 105:120) = 1; 
  Full_LK_alg(i1, i2, 0, mask, 5)
