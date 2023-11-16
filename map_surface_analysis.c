// Author: APD team, except where source was noted

#include "helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#define CONTOUR_CONFIG_COUNT 16
#define FILENAME_MAX_SIZE 50
#define STEP 8
#define SIGMA 200
#define RESCALE_X 2048
#define RESCALE_Y 2048

#define CLAMP(v, min, max) \
    if (v < min)           \
    {                      \
        v = min;           \
    }                      \
    else if (v > max)      \
    {                      \
        v = max;           \
    }

pthread_barrier_t b;

// Definirea structurii

typedef struct
{
    ppm_image *image;
    ppm_image *new_image;
    int thread_id;
    int P;
    ppm_image **map;
    unsigned char **grid;
    uint8_t sample[3];

} ThreadData;

// Functia thread

void *thread_function(void *arg)
{
    // Paralelizarea map
    ThreadData threadData = *(ThreadData *)arg;
    int start_row = (threadData.thread_id * CONTOUR_CONFIG_COUNT) / threadData.P;
    int end_row = ((threadData.thread_id + 1) * CONTOUR_CONFIG_COUNT) / threadData.P;
    for (int i = start_row; i < end_row; i++)
    {
        char filename[FILENAME_MAX_SIZE];
        sprintf(filename, "./contours/%d.ppm", i);
        threadData.map[i] = read_ppm(filename);
    }

    // Paralelizarea rescale
    if (threadData.image->x > RESCALE_X || threadData.image->y > RESCALE_Y)
    {
        threadData.new_image->x = RESCALE_X;
        threadData.new_image->y = RESCALE_Y;
        // use bicubic interpolation for scaling
        int start_x = (threadData.thread_id * RESCALE_X) / threadData.P;
        int end_x = ((threadData.thread_id + 1) * RESCALE_X) / threadData.P;

        for (int i = start_x; i < end_x; i++)
        {
            for (int j = 0; j < threadData.new_image->y; j++)
            {

                float u = (float)i / (float)(threadData.new_image->x - 1);
                float v = (float)j / (float)(threadData.new_image->y - 1);
                sample_bicubic(threadData.image, u, v, threadData.sample);

                threadData.new_image->data[i * threadData.new_image->y + j].red = threadData.sample[0];
                threadData.new_image->data[i * threadData.new_image->y + j].green = threadData.sample[1];
                threadData.new_image->data[i * threadData.new_image->y + j].blue = threadData.sample[2];
            }
        }
    }
    
    // Folosim bariera pentru a astepta threadurile 
    // Avem nevoie de infromatii care inca nu au fost calculate

    pthread_barrier_wait(&b);
    // grid

    int p = threadData.new_image->x / STEP;
    int q = threadData.new_image->y / STEP;

    int start_alloc_grid = (threadData.thread_id * p) / threadData.P;
    int end_alloc_grid = ((threadData.thread_id + 1) * p) / threadData.P;
    for (int i = start_alloc_grid; i < end_alloc_grid; i++)
    {
        for (int j = 0; j < q; j++)
        {
            ppm_pixel curr_pixel = threadData.new_image->data[i * STEP * threadData.new_image->y + j * STEP];

            unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

            if (curr_color > SIGMA)
            {
                threadData.grid[i][j] = 0;
            }
            else
            {
                threadData.grid[i][j] = 1;
            }
        }
    }
    for (int i = start_alloc_grid; i < end_alloc_grid; i++)
    {
        ppm_pixel curr_pixel = threadData.new_image->data[i * STEP * threadData.new_image->y + threadData.new_image->x - 1];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;

        if (curr_color > SIGMA)
        {
            threadData.grid[i][q] = 0;
        }
        else
        {
            threadData.grid[i][q] = 1;
        }
    }
    int start_alloc_grid_q = (threadData.thread_id * q) / threadData.P;
    int end_alloc_grid_q = ((threadData.thread_id + 1) * q) / threadData.P;

    for (int j = start_alloc_grid_q; j < end_alloc_grid_q; j++)
    {
        ppm_pixel curr_pixel = threadData.new_image->data[(threadData.new_image->x - 1) * threadData.new_image->y + j * STEP];

        unsigned char curr_color = (curr_pixel.red + curr_pixel.green + curr_pixel.blue) / 3;
        if (curr_color > SIGMA)
        {
            threadData.grid[p][j] = 0;
        }
        else
        {
            threadData.grid[p][j] = 1;
        }
    }
    
    // La fel ca mai sus folosim bariera, in continuare avem nevoie de informatii care
    // inca nu au fost calculate

    pthread_barrier_wait(&b);
    // march

    for (int i = start_alloc_grid; i < end_alloc_grid; i++)
    {
        for (int j = 0; j < q; j++)
        {
            unsigned char k = 8 * threadData.grid[i][j] + 4 * threadData.grid[i][j + 1] + 2 * threadData.grid[i + 1][j + 1] + 1 * threadData.grid[i + 1][j];

            // update image

            for (int l = 0; l < threadData.map[k]->x; l++)
            {
                for (int m = 0; m < threadData.map[k]->y; m++)
                {
                    int contour_pixel_index = threadData.map[k]->x * l + m;
                    int image_pixel_index = (i * STEP + l) * threadData.new_image->y + (j * STEP) + m;

                    threadData.new_image->data[image_pixel_index].red = threadData.map[k]->data[contour_pixel_index].red;
                    threadData.new_image->data[image_pixel_index].green = threadData.map[k]->data[contour_pixel_index].green;
                    threadData.new_image->data[image_pixel_index].blue = threadData.map[k]->data[contour_pixel_index].blue;
                }
            }
        }
    }
    pthread_exit(NULL);
}

// Calls `free` method on the utilized resources.
void free_resources(ppm_image *image, ppm_image **contour_map, unsigned char **grid, int step_x)
{
    for (int i = 0; i < CONTOUR_CONFIG_COUNT; i++)
    {
        free(contour_map[i]->data);
        free(contour_map[i]);
    }
    free(contour_map);

    for (int i = 0; i <= image->x / step_x; i++)
    {
        free(grid[i]);
    }
    free(grid);

    free(image->data);
    free(image);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        fprintf(stderr, "Usage: ./tema1 <in_file> <out_file> <P>\n");
        return 1;
    }
    
    // initializari si alocari de memorie

    ppm_image *image = read_ppm(argv[1]);

    int P = atoi(argv[3]);

    pthread_t tid[P];
    ThreadData threadData[P];

    ppm_image *new_image = (ppm_image *)malloc(sizeof(ppm_image));
    ppm_image **map = (ppm_image **)malloc(CONTOUR_CONFIG_COUNT * sizeof(ppm_image *));
    int p = 0;
    int q = 0;

    // Verificare daca imaginea primita ca argument trebuie rescalata
    // si atribuire de memorie conforma pentru new_image->data si grid

    if (image->x > RESCALE_X || image->y > RESCALE_Y)
    {
        new_image->data = (ppm_pixel *)malloc(RESCALE_X * RESCALE_Y * sizeof(ppm_pixel));
        new_image->x = RESCALE_X;
        new_image->y = RESCALE_Y;
        p = RESCALE_X / STEP;
        q = RESCALE_Y / STEP;
    }
    else
    {
        new_image->data = (ppm_pixel *)malloc(image->x * image->y * sizeof(ppm_pixel));
        new_image->data = image->data;
        new_image->x = image->x;
        new_image->y = image->y;
        new_image = image;
        p = image->x / STEP;
        q = image->y / STEP;
    }

    unsigned char **grid = (unsigned char **)malloc((p + 1) * sizeof(unsigned char *));
    if (!grid)
    {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    for (int i = 0; i <= p; i++)
    {
        grid[i] = (unsigned char *)malloc((q + 1) * sizeof(unsigned char));
        if (!grid[i])
        {
            fprintf(stderr, "Unable to allocate memory\n");
            exit(1);
        }
    }

    pthread_barrier_init(&b, NULL, P);

    // Modul in care am invatat sa folosim threadurile, si toate datele despre
    // imagini pasate intr un struct

    for (int i = 0; i < P; i++)
    {
        threadData[i].image = image;
        threadData[i].thread_id = i;
        threadData[i].P = P;
        threadData[i].grid = grid;
        threadData[i].map = map;
        threadData[i].new_image = new_image;
        pthread_create(&tid[i], NULL, thread_function, &threadData[i]);
    }

    for (int i = 0; i < P; i++)
    {
        pthread_join(tid[i], NULL);
    }

    write_ppm(new_image, argv[2]);
    free_resources(new_image, map, grid, STEP);
    free(image->data);
    free(image);

    return 0;
}
