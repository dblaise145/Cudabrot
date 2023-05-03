CC = g++
CFLAGS = -c

all: mandelbrot_cuda


mandelbrot_cuda: mandelbrot_cuda.cu
	nvcc mandelbrot_cuda.cu -c --expt-relaxed-constexpr -I SFML/include
	nvcc mandelbrot_cuda.obj -o mandel_cuda -L SFML/lib -lsfml-graphics -lsfml-window -lsfml-system