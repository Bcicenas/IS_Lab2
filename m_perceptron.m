x = linspace(0, 1, 20);

%plot curve
d = 0.6 * sin(2*pi*x) + cos(2*pi*x);
plot(x, d);
xlabel('x');
ylabel('d');

%Inputs for perceptron
n = 4; %four neurons
W = randn(n*2,1);
B = randn(n*2,1);
b_out = randn(1); % ouput bias value
y = zeros(1, 20);
e = zeros(1, 20);
nu = 0.2; %learning rate

for i=1:20
    %calculate output for the first layer
    y1_1 = 1 / 1 + exp(-(x(i) * W(1) + B(1)));
    y1_2 = 1 / 1 + exp(-(x(i) * W(2) + B(2)));
    y1_3 = 1 / 1 + exp(-(x(i) * W(3) + B(3)));
    y1_4 = 1 / 1 + exp(-(x(i) * W(4) + B(4)));
    
    %calculate ouput for the output layer
    y(i) = y1_1*W(5) + y1_2*W(6) + y1_3*W(7) + y1_4*W(8) + b_out;
    %calculate error
    e(i) = d(i) - y(i);
    
    %update weights for output layer
    W(5) = W(5) * nu * () * y1_1;
    W(6) = W(6) * nu * () * y1_2;
    W(7) = W(7) * nu * () * y1_3;
    W(8) = W(8) * nu * () * y1_4;
    
    %update weights for hidden layer
    
    
end

hold on;
plot(x, d*2);


