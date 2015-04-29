function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


%     %Y(1:10)
%     theta(1) = theta(1) - alpha*sum((Y - y).*X(:,1))/m;
%     theta(2) = theta(2) - alpha*sum((Y - y).*X(:,2))/m;
%     theta(3) = theta(3) - alpha*sum((Y - y).*X(:,3))/m;
    Y = X * theta;
    for i = 1:size(theta, 1)
        theta(i) = theta(i) - alpha*sum((Y - y).*X(:,i))/m; 
    end;

   % theta




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
