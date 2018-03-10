function [test_acc,test_confmat]=test_lr(w,testX,testt)
x_size=size(testX,1);
y=ones(x_size,1);
biasd_points=[testX y];
    posterior=sigmoid(biasd_points*w);
for i=1:x_size
    if(posterior(i)>(1-posterior(i)))
        y(i)=1;
    else
        y(i)=0;
    end
end
test_confmat = confusionmat(y,testt);
test_acc=((test_confmat(1,1)+test_confmat(2,2))/x_size)*100;
end