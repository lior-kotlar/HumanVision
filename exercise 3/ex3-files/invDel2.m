function [invK]=invDel2(im_size)

isize = 2*max(im_size);
 K=zeros(isize);
 K(isize/2,isize/2)=-4;
 K(isize/2+1,isize/2)=1;
 K(isize/2,isize/2+1)=1;
 K(isize/2-1,isize/2)=1;
 K(isize/2,isize/2-1)=1;
 
 Khat=fft2(K/4);
 I=find(Khat==0);
 Khat(I)=1;
 invKhat=1./Khat;
 invKhat(I)=0;
 invK=ifft2(invKhat);
 invK=real(invK);
 invK=conv2(invK,[1 0 0;0 0 0;0 0 0],'same');% shift by one
 