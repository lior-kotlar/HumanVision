function [im1,im2]=rhombusMovie(fatFlag,contrast)

im1=rhombusImage(128,0,fatFlag,contrast);
im2=rhombusImage(128,1,fatFlag,contrast);


function [im]=rhombusImage(isize,xShift,fatFlag,contrast)



if (fatFlag==1)
theta1=0;
theta2=45;
else
 theta1=30;
 theta2=45;
end



r=isize/6;

theta1=theta1*pi/180;
theta2=theta2*pi/180;

[x,y]=meshgrid(1:isize);x=x-isize/2;y=y-isize/2;
x=x-xShift;
xW=sin(theta1)*x+cos(theta1)*y;
yW=sin(theta2)*x+cos(theta2)*y;

M=[cos(theta1) sin(theta1);cos(theta2) sin(theta2)];
%r=r*abs(det(M));

%im=1-softThresh(max(abs(xW),abs(yW)),r*abs(det(M)),2);
im=1-softThresh(max(abs(xW),abs(yW)),r*abs(det(M)),1);
im=im*contrast;

function [y]=softThresh(x,t,s)
% a soft threshold function useful in many places in life
% t is the threshold
% s is softness
if ~exist('s')
  s=5;
end
y=1./(1+exp(-s*(x-t)));


