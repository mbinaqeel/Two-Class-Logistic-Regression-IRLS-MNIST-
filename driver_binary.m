load Data\mnist_uint8.mat
train_x=double(train_x);
train_y=double(train_y);
test_x=double(test_x);
test_y=double(test_y);

class1=4;
class2=9;

%training data
class1_inds=find(train_y(:,class1)==1);
class2_inds=find(train_y(:,class2)==1);
n1=length(class1_inds);
n2=length(class2_inds);
n=n1+n2;
training_inds=[class1_inds ; class2_inds];
X=double(train_x(training_inds,:));
t=[zeros(n1,1) ; ones(n2,1)];

%testing data
class1_inds=find(test_y(:,class1)==1);
class2_inds=find(test_y(:,class2)==1);
n1=length(class1_inds);
n2=length(class2_inds);
testn=n1+n2;
testing_inds=[class1_inds ; class2_inds];
testX=double(test_x(testing_inds,:));
testt=[zeros(n1,1) ; ones(n2,1)];

%train
[w,train_acc,train_confmat]=train_lr(X,t);

%test
[test_acc,test_confmat]=test_lr(w,testX,testt);
save lr_results train_acc train_confmat test_acc test_confmat