# matlab_rust

Some matlab functions implemented in rust, aiming to be used on `x86_64` CPUs.

So no GPU computing, just threading and SIMD.

Shout out to @RoyiAvital and @ZR Han, who gave amazing insights and Matlab code examples as references.

https://dsp.stackexchange.com/questions/74803/replicate-matlabs-conv2-in-frequency-domain

https://dsp.stackexchange.com/questions/38542/applying-image-filtering-circular-convolution-in-frequency-domain

## implemented functions

- conv2

- xcorr2

Which is just `conv2(a, rot180(b))`
