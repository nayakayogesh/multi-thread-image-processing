#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Configuration
#define BLUR_KERNEL_SIZE 3
#define SOBEL_SCALE 5.0f

// Global Variable
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

void save_grayscale_png(const char *filename, float **img) {
    unsigned char *data = malloc(width * height);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float v = img[y][x];
            if (v < 0) v = 0;
            if (v > 1) v = 1;
            data[y * width + x] = (unsigned char)(v * 255.0f + 0.5f);
        }
    }
    stbi_write_png(filename, width, height, 1, data, width);
    free(data);
    printf("Saved: %s\n", filename);
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


void stage_invert(float **input, float **output) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            output[y][x] = 1.0f - input[y][x];
}

void stage_blur(float **input, float **output) {
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };

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

    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        output[0][x] = input[0][x];
        output[height - 1][x] = input[height - 1][x];
    }
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        output[y][0] = input[y][0];
        output[y][width - 1] = input[y][width - 1];
    }
}

void stage_sobel(float **input, float **output) {
    float gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    float gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

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

    // Borders black
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (y == 0 || y == height-1 || x == 0 || x == width-1)
                output[y][x] = 0.0f;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s input_image.jpg\n", argv[0]);
        return 1;
    }

    omp_set_num_threads(8);

    const char *input_file = argv[1];
    printf("Loading: %s ...\n", input_file);

    float **grayscale = load_to_grayscale_float(input_file);
    float **inverted  = alloc_image();
    float **blurred   = alloc_image();
    float **edges     = alloc_image();

    double start = omp_get_wtime();

    stage_invert(grayscale, inverted);
    stage_blur(inverted, blurred);
    stage_sobel(blurred, edges);

    double end = omp_get_wtime();

    // Save stages
    save_grayscale_png("0_grayscale.png", grayscale);
    save_grayscale_png("1_inverted.png", inverted);
    save_grayscale_png("2_blurred.png", blurred);
    save_grayscale_png("3_edges.png", edges);

    printf("\nDone in %.3f seconds\n", end - start);
    printf("Generated files:\n");
    printf("  0_grayscale.png\n");
    printf("  1_inverted.png\n");
    printf("  2_blurred.png\n");
    printf("  3_edges.png\n");

    free_image(grayscale);
    free_image(inverted);
    free_image(blurred);
    free_image(edges);

    return 0;
}