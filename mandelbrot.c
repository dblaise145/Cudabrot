#include <complex.h>
#include <stdio.h>

#define MAX_ITER 10000

int mandelbrot(complex z, complex c)
{
    int i;
    for (i = 0; i < MAX_ITER; i++)
    {
        if (cabs(z) > 2.0)
            return i;
        z = z * z + c;
    }
    return MAX_ITER;
}

void generate_mandelbrot_set(int width, int height, double xmin, double xmax, double ymin, double ymax)
{
    int i, j;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            double x = xmin + j * (xmax - xmin) / (width - 1);
            double y = ymin + i * (ymax - ymin) / (height - 1);
            complex c = x + y * I;
            int value = mandelbrot(0, c);
            printf("%d ", value);
        }
        printf("\n");
    }
}

int main(void)
{
    int width = 80, height = 25;
    double xmin = -2.0, xmax = 1.0, ymin = -1.0, ymax = 1.0;
    generate_mandelbrot_set(width, height, xmin, xmax, ymin, ymax);
    return 0;
}