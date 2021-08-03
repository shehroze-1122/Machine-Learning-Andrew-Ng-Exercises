function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
% X= 10x3 
h = X*theta; %10x1
J = sum((h - y).^2);
J = J/(2*m);
reg = (lambda/(2*m))*sum(theta(2:end).^2);
J = J+reg;

grad(2:end,1) = (1/m)*X(:,2:end)'*(h-y) + (lambda/m)*theta(2:end);

grad(1,1) = (1/m)*X(:,1)'*(h-y);
% =========================================================================

grad = grad(:);

end
