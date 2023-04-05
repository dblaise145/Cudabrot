#include <SFML/Graphics.hpp>
#include <math.h>
#include <iostream>
#include <chrono>

__device__ void HSVtoRGB(int* red, int* green, int* blue, float H, float S, float V);
__device__ double mandelIter(double cx, double cy, int maxIter);
sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int percision, int iterations);
__global__ void mandel_kernel(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations, sf::Uint8* pixels);

int main()
{
	unsigned int width = 1600;
	unsigned int height = 900;

	sf::RenderWindow window(sf::VideoMode(width, height), "mandelbrot");

	window.setFramerateLimit(144);

	sf::Texture mandelTexture;
	sf::Sprite mandelSprite;



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

	int iterations = 1;
	int percision = 64;

	mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, percision, iterations);



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
					percision = 64;

					xmin = oxmin;
					xmax = oxmax;
					yRange = oyRange;
					ymin = oymin;
					ymax = oymax;
				}

				mandelTexture = mandelbrot(width, height, xmin, xmax, ymin, ymax, percision, iterations);
				break;
			case sf::Event::MouseWheelScrolled:
				if (evnt.mouseWheelScroll.delta <= 0)
				{
					percision /= 2;
					if (percision <= 4)
					{
						exit(0);
					}
				}
				else
				{
					percision *= 2;
				}
				mandelTexture = mandelbrot(width, height, xmin, xmax, ymin, ymax, percision, iterations);
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


sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int precision, int iterations)
{
	sf::Texture texture;
	texture.create(width, height);

	sf::Uint8* pixels;

	cudaMallocManaged(&pixels, sizeof(sf::Uint8) * (width * height * 4));

	auto start = std::chrono::high_resolution_clock::now();
	mandel_kernel << <4, 256 >> > (width, height, xmin, xmax, ymin, ymax, iterations, pixels);
	cudaDeviceSynchronize();
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
	printf("CUDA PREC: %d TIME: %8.4fs\n", precision, elapsed.count() * 1e-9);


	texture.update(pixels, width, height, 0, 0);

	cudaFree(pixels);

	return texture;
}


__global__
void mandel_kernel(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations, sf::Uint8* pixels)
{
	int i = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x +
		(blockIdx.x * blockDim.x + threadIdx.x);
	for (; i < width * height; i += blockDim.x * gridDim.x * blockDim.y * gridDim.y)
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
		int R;
		int G;
		int B;
		HSVtoRGB(&R, &G, &B, hue, sat, val);
		pixels[ppos] = B;
		pixels[ppos + 1] = G;
		pixels[ppos + 2] = R;
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
	float X = C * (1 - abs(fmod((double)(H / 60.0),(double) 2) - 1));
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