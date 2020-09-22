%create curve using trigonometry
% 1. create 20 points
x = linspace(0, 1, 20);

%plot curve
y = 0.6 * sin(2*pi*x) + cos(2*pi*x);
disp(y);
plot(x, y);
xlabel('x');
ylabel('y');

%create multilayer perceptron to approximate this curve
% 1 hidden layer with 4-8 neurons
n = 4;
B = randn(n, 1);
W = randn(n, 20, 1);
%1 step calculate output

Y = zeros(1, n);
E = zeros(1, n);

for i = 1:n
    v = sum( x(i:20) .* y(i:20)) + B(i);
    Y(i) = 1 / (1 + exp(-v));
    E(i) = sum(y) - Y(i);
end

disp(E);



