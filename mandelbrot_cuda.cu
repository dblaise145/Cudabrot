#include <SFML/Graphics.hpp>
#include <math.h>
#include <iostream>
#include "timer.h"

__device__ void HSVtoRGB(int* red, int* green, int* blue, float H, float S, float V);
__device__ double mandelIter(double cx, double cy, int maxIter);
double normalize(double value, double localMin, double localMax, double min, double max);
sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations);
sf::Texture julia(int width, int height, double cRe, double cIm, int iterations);
__global__ void mandel_kernel(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations, sf::Uint8* pixels);
__global__ void julia_kernel(int width, int height, double cRe, double cIm, int iterations, sf::Uint8* pixels);


bool makeJulia = true;

int main()
{
	unsigned int width = 1600;
	unsigned int height = 900;

	sf::RenderWindow window(sf::VideoMode(width, height), "mandelbrot");

	window.setFramerateLimit(144);

	sf::Texture mandelTexture;
	sf::Sprite mandelSprite;


  double cRe = -.7;
  double cIm = .27015;

	double oxmin = -2.4;
	double oxmax = 1.0;
	double oyRange = (oxmax - oxmin) * height / width;
	double oymin = -oyRange / 2;
	double oymax = oyRange / 2;

	double xmin = oxmin;
	double xmax = oxmax;
	double yRange = oyRange;
	double ymin = oymin;
	double ymax = oymax;

	int recLevel = 1;
	int precision = 512;

	if (makeJulia)
  {
    mandelTexture = julia(width, height, cRe, cIm, precision);
  }
  else{
    mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
  }



	while (window.isOpen())
	{
		sf::Event evnt;
		while (window.pollEvent(evnt))
		{
			switch (evnt.type)
			{
			case sf::Event::Closed:
				window.close();
				break;
			case sf::Event::KeyReleased:
				if (evnt.key.code == sf::Keyboard::Key::O)
				{
					recLevel = 1;
					precision = 64;

					xmin = oxmin;
					xmax = oxmax;
					yRange = oyRange;
					ymin = oymin;
					ymax = oymax;
				}
				if (makeJulia)
          {
            mandelTexture = julia(width, height, cRe, cIm, precision);
          }
        else
          {
            mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
          }
				break;
			case sf::Event::MouseWheelScrolled:
				if (evnt.mouseWheelScroll.delta <= 0)
				{
					precision /= 2;
          if (precision <= 4)
          {
            exit(0);
          }
				}
				else
				{
					precision *= 2;
				}
				if (makeJulia)
        {
          mandelTexture = julia(width, height, cRe, cIm, precision);
        }
        else
        {
          mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
        }
				break;
			}
		}

		mandelSprite.setTexture(mandelTexture);

		window.clear(sf::Color::White);

		window.draw(mandelSprite);

		window.display();
	}

	return 0;
}


double normalize(double value, double localMin, double localMax, double min, double max)
{
	double normalized = (value - localMin) / (localMax - localMin);
	normalized = normalized * (max - min);
	normalized += min;
	return normalized;
}

__device__
double mandelIter(double cx, double cy, int maxIter) {
	double x = 0.0;
	double y = 0.0;
	double xx = 0.0;
	double yy = 0.0;
	double xy = 0.0;

	double i = maxIter;
	while (i-- && xx + yy <= 4) {
		xy = x * y;
		xx = x * x;
		yy = y * y;
		x = xx - yy + cx;
		y = xy + xy + cy;
	}
	return maxIter - i;
}


sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int precision)
{
	sf::Texture texture;
	texture.create(width, height);

	sf::Uint8* pixels;

  cudaMallocManaged(&pixels, sizeof(sf::Uint8)*(width * height * 4));

  START_TIMER(prec);
  mandel_kernel<<<512, 512>>>(width, height, xmin, xmax, ymin, ymax, precision, pixels);
  cudaDeviceSynchronize();
  STOP_TIMER(prec);
  printf("PREC: %d TIME: %8.4fs\n", precision,  GET_TIMER(prec));


	texture.update(pixels, width, height, 0, 0);

	cudaFree(pixels);

	return texture;
}


sf::Texture julia(int width, int height, double cRe, double cIm, int iterations)
{
  sf::Texture texture;
  texture.create(width, height);

  sf::Uint8* pixels;

  cudaMallocManaged(&pixels, sizeof(sf::Uint8)*(width * height * 4));

  START_TIMER(prec);
  julia_kernel<<<512,512>>>(width, height, cRe, cIm, iterations, pixels);
  cudaDeviceSynchronize();
  STOP_TIMER(prec);
  printf("PREC: %d TIME: %8.4fs\n", iterations,  GET_TIMER(prec));

  texture.update(pixels, width, height, 0, 0);

	cudaFree(pixels);

	return texture;
}


__global__
void julia_kernel(int width, int height, double cRe, double cIm, int iterations, sf::Uint8* pixels)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < width * height; i += blockDim.x * gridDim.x)
	{
      int row = i / width;
      int col = i % width;
      int ppos = 4 * (width * row + col);
      double zx = 1.5 * (col - width / 2) / (.5 * width);
      double zy = (row - height / 2) / (.5 * height);

      int i;

      for (i = 0; i < iterations; i++)
      {
        double oldzx = zx;
        double oldzy = zy;

        zx = oldzx * oldzx - oldzy * oldzy + cRe;
        zy = 2 * oldzx * oldzy + cIm;

        if((zx * zx + zy * zy) > 4) break;
      }
      int R, G, B;
      HSVtoRGB(&R, &G, &B, (int)(255 * i / iterations), 100, (i > iterations) ? 0 : 100);
      pixels[ppos] = B;
			pixels[ppos + 1] = G;
			pixels[ppos + 2] = G * 2;
			pixels[ppos + 3] = 255;
  }
}


__global__
void mandel_kernel(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations, sf::Uint8* pixels)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < width * height; i += blockDim.x * gridDim.x)
	{
    int row = i / width;
    int col = i % width;
    double x = xmin + (xmax - xmin) * col / (width - 1.0);
    double y = ymin + (ymax - ymin) * row / (height - 1.0);

    double i = mandelIter(x, y, iterations);

    int ppos = 4 * (width * row + col);

    int hue = (int)(255 * i / iterations);
    int sat = 100;
    int val = (i > iterations) ? 0 : 100;
    int R, G, B;
    HSVtoRGB(&R, &G, &B, hue, sat, val);
    pixels[ppos] = B;
    pixels[ppos + 1] = G;
    pixels[ppos + 2] = G * 2;
    pixels[ppos + 3] = 255;
	}
}

__device__
void HSVtoRGB(int* red, int* green, int* blue, float H, float S, float V) 
{
	if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
    *red = 0;
    *green = 0;
    *blue = 0;
    return;
	}
	float s = S / 100;
	float v = V / 100;
	float C = s * v;
	float X = C * (1 - abs(fmodf(H / 60.0, 2) - 1));
	float m = v - C;
	float r, g, b;
	if (H >= 0 && H < 60) {
		r = C, g = X, b = 0;
	}
	else if (H >= 60 && H < 120) {
		r = X, g = C, b = 0;
	}
	else if (H >= 120 && H < 180) {
		r = 0, g = C, b = X;
	}
	else if (H >= 180 && H < 240) {
		r = 0, g = X, b = C;
	}
	else if (H >= 240 && H < 300) {
		r = X, g = 0, b = C;
	}
	else {
		r = C, g = 0, b = X;
	}
	int R = (r + m) * 255;
	int G = (g + m) * 255;
	int B = (b + m) * 255;
  *red = R;
  *green = G;
  *blue = B;
}
