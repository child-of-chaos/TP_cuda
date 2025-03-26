#include <iostream>
#include <cmath>
#include <fstream>
#include <cstring>

__device__ float distance_spatiale(int x1, int y1, int x2, int y2) {
    return sqrtf((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

__device__ float difference_intensite(unsigned char* image, int largeur, int hauteur, int x1, int y1, int x2, int y2) {
    return fabsf(image[y1 * largeur + x1] - image[y2 * largeur + x2]);
}

__global__ void appliquer_filtre_bilateral(unsigned char* image, unsigned char* image_filtrée, int largeur, int hauteur, float sigma_spatial, float sigma_intensite) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < largeur && y < hauteur) {
        float somme_poids = 0.0f;
        float somme_pixels = 0.0f;

        // Chaque thread calcule un poids pour un voisin
        int i = threadIdx.x - 3; // Les indices de la fenêtre sont de -3 à +3 (7x7 voisins)
        int j = threadIdx.y - 3;

        if (i >= -3 && i <= 3 && j >= -3 && j <= 3) {
            int x_voisin = x + i;
            int y_voisin = y + j;

            if (x_voisin >= 0 && x_voisin < largeur && y_voisin >= 0 && y_voisin < hauteur) {
                // Calcul du poids spatial et du poids d'intensité
                float poids_spatial = expf(-0.5f * (distance_spatiale(x, y, x_voisin, y_voisin) / sigma_spatial) * (distance_spatiale(x, y, x_voisin, y_voisin) / sigma_spatial));
                float poids_intensite = expf(-0.5f * (difference_intensite(image, largeur, hauteur, x, y, x_voisin, y_voisin) / sigma_intensite) * (difference_intensite(image, largeur, hauteur, x, y, x_voisin, y_voisin) / sigma_intensite));
                float poids = poids_spatial * poids_intensite;

                // Accumulation des poids et des pixels (en utilisant les variables locales dans chaque thread)
                somme_poids += poids; 
                somme_pixels += image[y_voisin * largeur + x_voisin] * poids; 
            }
        }

        // Si tous les voisins ont été traités, calcul du pixel filtré dans le thread central
        if (threadIdx.x == 3 && threadIdx.y == 3) {
            image_filtrée[y * largeur + x] = (unsigned char)(somme_pixels / somme_poids);
        }
    }
}

void ecrire_BMP(const char* chemin, unsigned char* image, int largeur, int hauteur) {
    FILE* fichier = fopen(chemin, "wb");
    if (!fichier) {
        std::cerr << "Erreur d'ouverture du fichier !" << std::endl;
        return;
    }

    unsigned char header[54] = {'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 1, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    unsigned int fileSize = 54 + 256 * 4 + largeur * hauteur;
    *(unsigned int*)&header[2] = fileSize;
    *(unsigned int*)&header[18] = largeur;
    *(unsigned int*)&header[22] = hauteur;
    *(unsigned int*)&header[34] = 256; // Nombre de couleurs
    *(unsigned int*)&header[38] = 2835; // Résolution X
    *(unsigned int*)&header[42] = 2835; // Résolution Y

    fwrite(header, 1, 54, fichier);

    for (int i = 0; i < largeur * hauteur; ++i) {
        unsigned char couleur[4] = {image[i], image[i], image[i], 0};
        fwrite(couleur, 1, 4, fichier);
    }

    fclose(fichier);
}

void charger_image(const char* chemin, unsigned char* image, int largeur, int hauteur) {
    FILE* fichier = fopen(chemin, "rb");
    if (!fichier) {
        std::cerr << "Erreur d'ouverture du fichier image !" << std::endl;
        return;
    }

    fseek(fichier, 54, SEEK_SET);  // Sauter l'en-tête BMP
    fread(image, sizeof(unsigned char), largeur * hauteur, fichier);
    fclose(fichier);
}

int main() {
    const int largeur = 512;
    const int hauteur = 512;
    unsigned char* image = new unsigned char[largeur * hauteur];
    unsigned char* image_filtrée = new unsigned char[largeur * hauteur];

    // Charger l'image "lena_gray.bmp" depuis le dossier "/workspace/TP_cuda/"
    charger_image("/workspace/TP_cuda/lena_gray.bmp", image, largeur, hauteur);

    // Appliquer le filtre bilatéral
    unsigned char* d_image;
    unsigned char* d_image_filtrée;
    cudaMalloc(&d_image, largeur * hauteur * sizeof(unsigned char));
    cudaMalloc(&d_image_filtrée, largeur * hauteur * sizeof(unsigned char));

    cudaMemcpy(d_image, image, largeur * hauteur * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(7, 7); // Chaque thread dans la fenêtre de 7x7
    dim3 numBlocks((largeur + 6) / 7, (hauteur + 6) / 7); // Nombre de blocs pour couvrir l'image
    appliquer_filtre_bilateral<<<numBlocks, threadsPerBlock>>>(d_image, d_image_filtrée, largeur, hauteur, 1.0f, 50.0f);

    cudaMemcpy(image_filtrée, d_image_filtrée, largeur * hauteur * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Sauvegarder l'image filtrée dans un fichier BMP
    ecrire_BMP("image_filtree.bmp", image_filtrée, largeur, hauteur);

    // Libérer la mémoire
    delete[] image;
    delete[] image_filtrée;
    cudaFree(d_image);
    cudaFree(d_image_filtrée);

    return 0;
}

