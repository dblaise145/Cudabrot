CC = gcc
CFLAGS = -std=c99

mandelbrot: mandelbrot.c
	$(CC) $(CFLAGS) mandelbrot.c -o mandelbrot

clean:
	rm -f mandelbrot