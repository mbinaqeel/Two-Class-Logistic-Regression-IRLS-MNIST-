function [result] = sigmoid(a)
%computes sigmoid
    result = 1./(1+exp(-a));
end