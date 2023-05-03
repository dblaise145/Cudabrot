#include <SFML/Graphics.hpp>
#include <math.h>
#include <iostream>
#include "timer.h"
#include <stdio.h>
#include <string.h>
sf::Color HSVtoRGB(float H, float S, float V);
double normalize(double value, double localMin, double localMax, double min, double max);
double mandelIter(double cx, double cy, int maxIter);
sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations);
sf::Texture julia(int width, int height, double cRe, double cIm, int iterations);
sf::Texture transform_pixels(int width, int height);
int transform_count = 0;
bool makeJulia = true;
sf::Uint8* current_pixels;
int main()
{
	unsigned int width = 1600; 
	unsigned int height = 900;
	current_pixels = new sf::Uint8[width * height * 4];
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
	              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
  }
  else{
    mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
	              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
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
					precision = 128;

					xmin = oxmin;
					xmax = oxmax;
					yRange = oyRange;
					ymin = oymin;
					ymax = oymax;
				}
				else if (evnt.key.code == sf::Keyboard::Key::T) 
				{
					mandelTexture = transform_pixels(width, height);
					transform_count = (transform_count + 1) % 3;
					break;
				}
        else if (evnt.key.code == sf::Keyboard::Key::A)
        {
          if(makeJulia)
          {
            cRe-=.01;
            cIm-=.01;
            mandelTexture = julia(width, height, cRe, cIm, precision);
			
          }
        }
        else if (evnt.key.code == sf::Keyboard::Key::D)
        {
          if(makeJulia)
          {
            cRe+=.01;
            cIm+=.01;
            mandelTexture = julia(width, height, cRe, cIm, precision);
          }
        }
        else if (evnt.key.code == sf::Keyboard::Key::W)
        {
          if(makeJulia)
          {
            cRe+=.01;
            cIm-=.01;
            mandelTexture = julia(width, height, cRe, cIm, precision);
          }
        }
        else if (evnt.key.code == sf::Keyboard::Key::S)
        {
          if(makeJulia)
          {
            cRe-=.01;
            cIm+=.01;
            mandelTexture = julia(width, height, cRe, cIm, precision);
          }
        }
        else if (evnt.key.code == sf::Keyboard::Key::J)
        {
          if(makeJulia)
          {
            makeJulia = false;
            mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
			              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
          }
          else
          {
            makeJulia = true;
            mandelTexture = julia(width, height, cRe, cIm, precision);
			              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
          }
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
		  			              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
        }
        else
        {
          mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax, precision);
		  			              for(int i = 0; i < transform_count; i++)
              {
                mandelTexture = transform_pixels(width, height);
              }
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

sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax, int iterations)
{
	sf::Texture texture;
	texture.create(width, height);

	sf::Uint8* pixels = new sf::Uint8[width * height * 4];

  START_TIMER(prec);
	for (int ix = 0; ix < width; ix++)
	{
		for (int iy = 0; iy < height; iy++)
		{
			double x = xmin + (xmax - xmin) * ix / (width - 1.0);
			double y = ymin + (ymax - ymin) * iy / (height - 1.0);

			double i = mandelIter(x, y, iterations);

			int ppos = 4 * (width * iy + ix);

			int hue = (int)(255 * i / iterations);
			int sat = 100;
			int val = (i > iterations) ? 0 : 100;
			sf::Color hsvtorgb = HSVtoRGB(hue, sat, val);
			pixels[ppos] = (int)hsvtorgb.b;
			pixels[ppos + 1] = (int)hsvtorgb.g;
			pixels[ppos + 2] = (int)hsvtorgb.g * 2;
			pixels[ppos + 3] = 255;
		}
	}
  STOP_TIMER(prec);
  printf("PREC: %d TIME: %8.4fs\n", iterations,  GET_TIMER(prec));

	texture.update(pixels, width, height, 0, 0);
	memcpy(current_pixels, pixels, width * height * 4 );
	delete[] pixels;

	return texture;
}

sf::Texture transform_pixels(int width, int height) {
	START_TIMER(prec);
	for (int ix = 0; ix < width; ix++)
  {
    for (int iy = 0; iy < height; iy++)
    {
		      int ppos = 4 * (width * iy + ix);
 	  sf::Uint8 tmp_red = current_pixels[ppos];
      sf::Uint8 tmp_blue = current_pixels[ppos + 1];
      sf::Uint8 tmp_green = current_pixels[ppos + 2];
      current_pixels[ppos] = tmp_blue;
      current_pixels[ppos + 1] = tmp_green;
      current_pixels[ppos + 2] = tmp_red;
	  current_pixels[ppos + 3] = 255;
    }
  }
  	STOP_TIMER(prec);
	printf("Transform TIME: %8.4fs\n", GET_TIMER(prec));
  	sf::Texture texture;
	texture.create(width, height);
	texture.update(current_pixels, width, height, 0, 0);
	return texture;
}


sf::Texture julia(int width, int height, double cRe, double cIm, int iterations)
{
  sf::Texture texture;
  texture.create(width, height);

  sf::Uint8* pixels = new sf::Uint8[width * height * 4];

  START_TIMER(prec);
  for (int ix = 0; ix < width; ix++)
  {
    for (int iy = 0; iy < height; iy++)
    {
      int ppos = 4 * (width * iy + ix);
      double zx = 1.5 * (ix - width / 2) / (.5 * width);
      double zy = (iy - height / 2) / (.5 * height);

      int i;

      for (i = 0; i < iterations; i++)
      {
        double oldzx = zx;
        double oldzy = zy;

        zx = oldzx * oldzx - oldzy * oldzy + cRe;
        zy = 2 * oldzx * oldzy + cIm;

        if((zx * zx + zy * zy) > 4) break;
      }
      sf::Color hsvtorgb = HSVtoRGB((int)(255 * i / iterations), 100, (i > iterations) ? 0 : 100);
      pixels[ppos] = (int)hsvtorgb.b;
			pixels[ppos + 1] = (int)hsvtorgb.g;
			pixels[ppos + 2] = (int)hsvtorgb.g * 2;
			pixels[ppos + 3] = 255;
    }
  }
  STOP_TIMER(prec);
  printf("PREC: %d TIME: %8.4fs\n", iterations,  GET_TIMER(prec));
  texture.update(pixels, width, height, 0, 0);
	memcpy(current_pixels, pixels, width * height * 4 );
	delete[] pixels;

	return texture;
}

sf::Color HSVtoRGB(float H, float S, float V) 
{
	if (H > 360 || H < 0 || S>100 || S < 0 || V>100 || V < 0) {
		return sf::Color::Black;
	}
	float s = S / 100;
	float v = V / 100;
	float C = s * v;
	float X = C * (1 - abs(fmod(H / 60.0, 2) - 1));
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
	return sf::Color(R, G, B);
}
