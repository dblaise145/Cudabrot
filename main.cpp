#include <SFML/Graphics.hpp>
#include <complex.h>
#include <stdio.h>


sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax);

int main()
{
    unsigned int width = 800;
    unsigned int height = 600;
    sf::RenderWindow window(sf::VideoMode(width, height), "SFML works!");
    window.setFramerateLimit(60);

    sf::RectangleShape zoomBorder(sf::Vector2f(width / 8, height / 8));
    zoomBorder.setFillColor(sf::Color(0, 0, 0, 0));
    zoomBorder.setOutlineColor(sf::Color(255, 255, 255, 128));
    zoomBorder.setOutlineThickness(1.0f);
    zoomBorder.setOrigin(sf::Vector2f(zoomBorder.getSize().x / 2, zoomBorder.getSize().y / 2));




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


    int level = 1;
    int precision = 64;

    sf::Texture mandelTexture = mandelbrot(width, height, oxmin, oxmax, oymin, oymax);
	  sf::Sprite mandelSprite;


    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.display();
    }



    return 0;
}

int mandelIter(complex z, complex c, int iter)
{
    int i;
    for (i = iter; i < MAX_ITER; i++)
    {
        if (cabs(z) > 2.0)
            return i;
        z = z * z + c;
    }
    return MAX_ITER;
}


sf::Texture mandelbrot(int width, int height, double xmin, double xmax, double ymin, double ymax)
{
  sf::Texture texture;
  texture.create(width, height);

  sf::Uint8* pixels = new sf::Uint8[width * height * 4];

  int i, j;
  for (i = 0; i < height; i++)
  {
      for (j = 0; j < width; j++)
      {
          double x = xmin + j * (xmax - xmin) / (width - 1);
          double y = ymin + i * (ymax - ymin) / (height - 1);

          int pixel = 4 *(width * j + i);


          complex c = x + y * I;
          int value = mandelbrot(0, c);
          printf("%d ", value);
      }
      printf("\n");
  }

  
}