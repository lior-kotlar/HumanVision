function [M]=mymovie(im1,im2,r_flag)

if nargin<3
    r_flag = 0;
end

[sx,sy]=size(im1);
seq=zeros(sx,sy,2);
seq(:,:,1)=im1;
seq(:,:,2)=im2;
nframes=2;

show(seq(:,:,1));
M=moviein(nframes);

for i=1:nframes
   figure(1);hold off;
   if r_flag
    imshow(seq(:,:,i));
   else
    show(seq(:,:,i));   
   end
   M(:,i)=getframe;
end

movie(M,-20);


