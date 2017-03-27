#include "fft.hpp"

/*
This software is dual-licensed to the public domain and under the following
license: you are granted a perpetual, irrevocable license to copy, modify,
publish, and distribute this file as you see fit.
THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
*/
 
#include <stdint.h>
#include <stdio.h>
#include <math.h> // cos
#include <vector> // lazy
#include <String.h> // strerror

// "Hann" window. This is essentially just the shape of a cosine
// from 0.0 to 1.0 to 0.0. The difference from a cosine window is
// that the Hann window is completely interpolant (jargon meaning
// that we don't have to normalize it when we overlap copies of it)
// because it's horizontally flat at the start, middle, and end. A
// real cosine window contains 50% the length of a single cycle of
// the cosine, not 100%.
/*   0         1
 1.0|   ....   |
    |  .    .  |
 0.0|..      ..|
    |          |
-1.0|          |
*/
inline double window_hann(double pace)
{
    return -cos(pace*M_PI*2)/2+0.5;
}
// 0 0 1 1 1 1 0 0 value
// 0 1 2 3 4 5 6 7 place
// span = 8
// 8/4 = 2
// 8*3/4 = 6
//  < 2
// >= 6
inline double window_box(int place, int span)
{
    if(place < span/4 or place >= span*3/4)
        return 0;
    else
        return 1;
}
// zero-padded hann window
inline double window(int place, int span)
{
    // our center point is span/2
    double pace = place/double(span);
    
    if(pace > 0.25 and pace < 0.75)
        return window_hann(pace*2-0.5);
    else
        return 0;
}

#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)
inline double get_sample(int16_t* v, size_t size, int64_t i)
{
    if(unlikely(i < 0 or i >= size))
        return 0.0;
    else
        return double(v[i])/double(32768);
}
inline void set_sample(double* v, size_t size, int64_t i, double n)
{
    if(likely(i >= 0 and i < size))
        v[i] = n;
}

// We need our filter to be zero-padded, because otherwise it 
void init_kernel(const int span, double * real_bins, double * imag_bins)
{
    double * real_samples = (double*)malloc(span*sizeof(double));
    double * imag_samples = (double*)malloc(span*sizeof(double));
    
    for(int i = 0; i < span; i++)
    {
        // for sinc, make x=0 at span/2 (indeed, the center of a DFT is a single sample)
        double value;
        int int_x = i-span/2;
        if(int_x == 0)
            value = 1;
        else
        {
            double x = int_x;
            x *= M_PI; // "normalized" sinc
            x /= 2; // our frequency cutoff (nyquist/2)
            value = sin(x)/x;
        }
        value *= window(i, span)/2;
        value *= sqrt(span); // convolution in the spatial domain is equivalent to pointwise multiplication in the frequency domain given a constant
        
        // the "center" sample of an FFT is the very first one, therefore we have to wrap around the other side
        int index = (i+span/2)%span;
        real_samples[index] = value;
        imag_samples[index] = value;
    }
    
    // Step 3: Convert to frequency form
    fft(real_samples, imag_samples, span, real_bins, imag_bins);
    // Free temporary resources
    free(real_samples);
    free(imag_samples);
}
int main()
{
    // expects monaural signed 16-bit audio
    auto input = fopen("demo.pcm", "rb");
    auto output = fopen("output.pcm", "wb");
    
    if(!input or !output) return puts("Error opening a file"), 0;
    
    /*
    Convolution in the spatial domain takes O(n) operations to produce n samples.
    
    However, convolution in the spatial domain is equivalent to doing
    pointwise multiplication in the frequency domain.
    
    Doing an FFT on n samples requires O(nlogn) operations.
    You get O(n) samples out of doing convolution with FFTs.
    
    FFTs only work with cyclical data (think a looping waveform), so when we
    split up continuous data to make FFTs with it, we need to zero-pad it and/or
    use a window, otherwise frequencies that don't start and end at exactly the
    loop point will cause distortion/clicking in the FFT. (zero-padding is
    better if you're going to convert back to the spatial domain, a window is
    better if you need to look at it on a spectrogram, audacity's spectrogram
    allows you to do both at the same time.)
    
    You know from convolution that if you have an A-sample-long kernel, then
    a B-sample-long waveform will only be extended by A samples on either side.
    Because of this, we know that if our filter is small enough in the spatial
    domain before transforming it to the frequency domain, we won't cause the
    convoluted signal to wrap around the entire length of the zero-padded chunk
    of audio. This is what this program does. This is mathematically identical
    to normal convolution, and doesn't cause any artefacts (my previous
    implementation didn't zero-pad 
    */
    
    fseek(input, 0, SEEK_END);
    auto filesize = ftell(input);
    fseek(input, 0, SEEK_SET);
    
    auto count_input_samples = filesize/sizeof(int16_t); // truncates intentionally
    int16_t * input_samples = (int16_t*)malloc(count_input_samples*sizeof(int16_t));
    
    fread(input_samples, sizeof(int16_t), count_input_samples, input);
    
    // stores ouput waveform in floating point format
    auto count_output_samples = count_input_samples;
    double * output_samples = (double*)malloc(count_output_samples*sizeof(double));
    
    const int quality = 512;
    const int span = quality*2; // size of the fft and its input data
    
    // temporary buffers
    char * heap = (char*)malloc(span*sizeof(double)*7);
    // put everything in a single contiguous heap to try to help the CPU cache
    double * windowed_samples = (double*)(heap+(span*sizeof(double)*0));
    double * real_bins        = (double*)(heap+(span*sizeof(double)*1));
    double * imag_bins        = (double*)(heap+(span*sizeof(double)*2));
    double * real_samples     = (double*)(heap+(span*sizeof(double)*3));
    double * imag_samples     = (double*)(heap+(span*sizeof(double)*4)); // don't ask...
    double * filter_real_bins = (double*)(heap+(span*sizeof(double)*5));
    double * filter_imag_bins = (double*)(heap+(span*sizeof(double)*6));
    
    init_kernel(span, filter_real_bins, filter_imag_bins);
    
    for(auto i = 0; i < count_input_samples; i += quality)
    {
        // copy input stream data into windowed buffer
        for(auto j = 0; j < span; j++) // starts at -quality from current sample "i"
        {
            double sample = get_sample(input_samples, count_input_samples, i+j-quality);
            windowed_samples[j] = sample * window_box(j, span);
        }
        
        // turn our spatial (waveform) data into frequency data
        fft(windowed_samples, nullptr, span, real_bins, imag_bins);
        
        // set all bins with a frequency above 50% of the nyquist frequency
        // (i.e. above 25% of the sample rate) to 0
        for(auto j = 0; j < span; j++)
        {
            real_bins[j] *= filter_real_bins[j];
            imag_bins[j] *= filter_imag_bins[j];
        }
        
        // turn our filtered frequency data into filtered spatial (waveform) data
        ifft(real_bins, imag_bins, span, real_samples, imag_samples);
        
        // Add our filtered samples to the output stream. Note that I chose the
        // Hann window specifically because it makes this process absolutely
        // trivial, and if you use a different window, you have to actually
        // ensure that the "lapping" between windowed chunks has a smooth shape.
        for(auto j = 0; j < span; j++)
        {
            //set_sample(output_samples, i+j-quality, real_samples[j]);
            int64_t pos = i+j-quality;
            if(pos >= 0 and pos < count_output_samples)
                output_samples[pos] += real_samples[j];
        }
    }
    
    for(auto i = 0; i < count_output_samples; i++)
    {
        // convert range clamp
        double n = output_samples[i]*double(32768);
        if(n >  32767) n =  32767;
        if(n < -32768) n = -32768;
        
        int16_t sample = round(n);
        
        fwrite(&sample, sizeof(int16_t), 1, output);
    }
}
