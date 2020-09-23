x = linspace(0, 1, 20);

%plot curve
d = 0.6 * sin(2*pi*x) + cos(2*pi*x);
plot(x, d);
xlabel('x');
ylabel('d');

%Inputs for perceptron
n = 6; %four neurons
W = randn(n*2,1);
B = randn(n,1);
b_out = randn(1); % ouput bias value
y = zeros(1, 20);
e = zeros(1, 20);
nu = 0.03; %learning rate

for l=1:1000000
    for i=1:20
        %calculate output for the first layer
        y1_1 = 1 / (1 + exp(-(x(i) * W(1) + B(1))));
        y1_2 = 1 / (1 + exp(-(x(i) * W(2) + B(2))));
        y1_3 = 1 / (1 + exp(-(x(i) * W(3) + B(3))));
        y1_4 = 1 / (1 + exp(-(x(i) * W(4) + B(4))));
        y1_5 = 1 / (1 + exp(-(x(i) * W(5) + B(5))));
        y1_6 = 1 / (1 + exp(-(x(i) * W(6) + B(6))));
        
        %calculate ouput for the output layer
        y(i) = y1_1*W(7) + y1_2*W(8) + y1_3*W(9) + y1_4*W(10) + y1_5*W(11) + y1_6*W(12) + b_out;
        %calculate error
        e(i) = d(i) - y(i);

        W(1) = W(1) + nu * (y1_1 * (1 - y1_1) * (e(i) * W(7))) * x(i);
        W(2) = W(2) + nu * (y1_2 * (1 - y1_2) * (e(i) * W(8))) * x(i);
        W(3) = W(3) + nu * (y1_3 * (1 - y1_3) * (e(i) * W(9))) * x(i);
        W(4) = W(4) + nu * (y1_4 * (1 - y1_4) * (e(i) * W(10))) * x(i);
        W(5) = W(5) + nu * (y1_5 * (1 - y1_5) * (e(i) * W(11))) * x(i);
        W(6) = W(6) + nu * (y1_6 * (1 - y1_6) * (e(i) * W(12))) * x(i);
        
        B(1) = B(1) + nu * (y1_1 * (1 - y1_1) * (e(i) * W(7)));
        B(2) = B(2) + nu * (y1_2 * (1 - y1_2) * (e(i) * W(8)));
        B(3) = B(3) + nu * (y1_3 * (1 - y1_3) * (e(i) * W(9)));
        B(4) = B(4) + nu * (y1_4 * (1 - y1_4) * (e(i) * W(10)));
        B(5) = B(5) + nu * (y1_5 * (1 - y1_5) * (e(i) * W(11)));
        B(6) = B(6) + nu * (y1_6 * (1 - y1_6) * (e(i) * W(12)));
        
        %update weights for output layer
        W(7) = W(7) + (nu * e(i) * y1_1);
        W(8) = W(8) + (nu * e(i) * y1_2);
        W(9) = W(9) + (nu * e(i) * y1_3);
        W(10) = W(10) + (nu * e(i) * y1_4);
        W(11) = W(11) + (nu * e(i) * y1_5);
        W(12) = W(12) + (nu * e(i) * y1_6);
        b_out = b_out + (nu * e(i));
        %update weights for hidden layer
 
    end
end
test = zeros(1, 20);
x1 = 0.1:1/22:1;
for i=1:20
    y1_1 = 1 / (1 + exp(-(x1(i) * W(1) + B(1))));
    y1_2 = 1 / (1 + exp(-(x1(i) * W(2) + B(2))));
    y1_3 = 1 / (1 + exp(-(x1(i) * W(3) + B(3))));
    y1_4 = 1 / (1 + exp(-(x1(i) * W(4) + B(4))));
    y1_5 = 1 / (1 + exp(-(x1(i) * W(5) + B(5))));
    y1_6 = 1 / (1 + exp(-(x1(i) * W(6) + B(6))));
    %calculate ouput for the output layer
    test(i) = y1_1*W(7) + y1_2*W(8) + y1_3*W(9) + y1_4*W(10) + y1_5*W(11) + y1_6*W(12) + b_out;
end
disp(W);
disp(test);
hold on;
plot(x, test);

