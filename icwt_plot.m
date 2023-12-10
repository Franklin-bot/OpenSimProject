load('/Users/FranklinZhao/OpenSimProject/data.mat')

fs = 200
wavelet = "morse";

[coeffs, f] = cwt(data, wavelet)
new_signal = icwt(coeffs, f, )
plot(new_signal)
hold on
plot(data)

