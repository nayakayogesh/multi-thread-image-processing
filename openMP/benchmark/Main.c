#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

// Configuration
#define BLUR_KERNEL_SIZE 3
#define SOBEL_SCALE      5.0f
#define REPEATS          3
#define MAX_THREADS      6

// Global Variables
int width = 0;
int height = 0;


float **load_to_grayscale_float(const char *filename) {
    int channels;
    unsigned char *data = stbi_load(filename, &width, &height, &channels, 0);
    if (!data) {
        fprintf(stderr, "Cannot load image: %s\n", filename);
        exit(1);
    }

    float **img = malloc(height * sizeof(float *));
    for (int y = 0; y < height; y++) {
        img[y] = malloc(width * sizeof(float));
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            float r = data[idx + 0] / 255.0f;
            float g = (channels >= 2) ? data[idx + 1] / 255.0f : r;
            float b = (channels >= 3) ? data[idx + 2] / 255.0f : r;
            img[y][x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }

    stbi_image_free(data);
    return img;
}

float **alloc_image() {
    float **img = malloc(height * sizeof(float *));
    for (int i = 0; i < height; i++)
        img[i] = malloc(width * sizeof(float));
    return img;
}

void free_image(float **img) {
    for (int i = 0; i < height; i++) free(img[i]);
    free(img);
}


void stage_invert(float **input, float **output, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y][x] = 1.0f - input[y][x];
}

void stage_blur(float **input, float **output, int num_threads) {
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++)
                    sum += input[y + ky][x + kx] * kernel[ky + 1][kx + 1];
            output[y][x] = sum;
        }
    }

    // Borders (serial – small work)
    for (int x = 0; x < width; x++) {
        output[0][x] = input[0][x];
        output[height - 1][x] = input[height - 1][x];
    }
    for (int y = 0; y < height; y++) {
        output[y][0] = input[y][0];
        output[y][width - 1] = input[y][width - 1];
    }
}

void stage_sobel(float **input, float **output, int num_threads) {
    float gx_kernel[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}};
    float gy_kernel[3][3] = {{-1,-2,-1}, {0,0,0}, {1,2,1}};

    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    float p = input[y + ky][x + kx];
                    gx += p * gx_kernel[ky + 1][kx + 1];
                    gy += p * gy_kernel[ky + 1][kx + 1];
                }
            }
            float mag = hypotf(gx, gy);
            output[y][x] = fminf(1.0f, mag / SOBEL_SCALE);
        }
    }

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (y == 0 || y == height-1 || x == 0 || x == width-1)
                output[y][x] = 0.0f;
}


double run_pipeline(float **grayscale, float **inverted, float **blurred, float **edges, int num_threads) {
    double start = omp_get_wtime();
    stage_invert(grayscale, inverted, num_threads);
    stage_blur(inverted, blurred, num_threads);
    stage_sobel(blurred, edges, num_threads);
    return omp_get_wtime() - start;
}


int main() {
    const char *test_files[] = {
        "../image_down_size/test_images/test_4k.jpg",
        "../image_down_size/test_images/test_1080p.jpg",
        "../image_down_size/test_images/test_720p.jpg",
        "../image_down_size/test_images/test_480p.jpg",
        "../image_down_size/test_images/test_240p.jpg",
        "../image_down_size/test_images/test_122p.jpg"
    };
    int num_tests = sizeof(test_files) / sizeof(test_files[0]);

    FILE *csv = fopen("benchmark_results.csv", "w");
    if (!csv) {
        perror("Cannot open benchmark_results.csv");
        return 1;
    }

    fprintf(csv, "image_resolution,threads,avg_time_seconds,speedup\n");

    for (int i = 0; i < num_tests; i++) {
        const char *filename = test_files[i];
        printf("\nBenchmarking %s ...\n", filename);

        float **grayscale = load_to_grayscale_float(filename);
        float **inverted  = alloc_image();
        float **blurred   = alloc_image();
        float **edges     = alloc_image();

        char resolution[32];
        snprintf(resolution, sizeof(resolution), "%dx%d", width, height);

        double seq_time = 0.0;

        for (int threads = 1; threads <= MAX_THREADS; threads++) {
            double total_time = 0.0;

            for (int rep = 0; rep < REPEATS; rep++) {
                total_time += run_pipeline(grayscale, inverted, blurred, edges, threads);
            }

            double avg_time = total_time / REPEATS;

            double speedup = (threads == 1) ? 1.0 : (seq_time / avg_time);

            if (threads == 1) {
                seq_time = avg_time;
            }

            fprintf(csv, "%s,%d,%.6f,%.3f\n",
                    resolution, threads, avg_time, speedup);

            printf("  Threads %2d:  %.4f s   (speedup %.2f×)\n",
                   threads, avg_time, speedup);
        }

        free_image(grayscale);
        free_image(inverted);
        free_image(blurred);
        free_image(edges);
    }

    fclose(csv);
    printf("\nResults saved to benchmark_results.csv\n");

    return 0;
}