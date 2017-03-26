demo.cpp
implements a lowpass at 50% the nyquist frequency using FFT

fft.hpp
implements DIT-2 FFT

demo.pcm
input audio file for convenience, 44100hz (excluded from repository)

Note that the output file is going to have minor clipping because lowpasses cause overshoot/ringing. The clipping is caused by converting the lowpassed signal to 16 bits with the same amplitude as the original file, not the lowpass process.
