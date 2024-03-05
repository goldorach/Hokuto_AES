#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <stdint.h> 





__device__ unsigned char gmul_table[256][256];

#define DEBUG_MODE  0

#define TIMER 0



__constant__ unsigned char const_sBox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};



__constant__ unsigned char const_invSBox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
};



__constant__ unsigned char const_rcon[10] = {
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
};



__device__ void KeyExpansion(unsigned char* key, unsigned char* roundKeys) {
    int i, j;
    unsigned char temp[4], k;

    for (i = 0; i < 16; i++) {
        roundKeys[i] = key[i];
    }

    for (; i < 176; i += 4) {
        for (j = 0; j < 4; j++) {
            temp[j] = roundKeys[i - 4 + j];
        }

        if (i % 16 == 0) {
            k = temp[0];
            temp[0] = temp[1];
            temp[1] = temp[2];
            temp[2] = temp[3];
            temp[3] = k;

            for (j = 0; j < 4; j++) {
                temp[j] = const_sBox[temp[j]];
            }

            temp[0] = temp[0] ^ const_rcon[i / 16 - 1];
        }

        for (j = 0; j < 4; j++) {
            roundKeys[i + j] = roundKeys[i - 16 + j] ^ temp[j];
        }
    }
}



__device__ void SubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = const_sBox[state[i]];
    }
}


__device__ void InvSubBytes(unsigned char* state) {
    for (int i = 0; i < 16; i++) {
        state[i] = const_invSBox[state[i]];
    }
}



__device__ void ShiftRows(unsigned char* state) {
    unsigned char temp;




    temp = state[1];
    state[1] = state[5];
    state[5] = state[9];
    state[9] = state[13];
    state[13] = temp;


    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;


    temp = state[3];
    state[3] = state[15];
    state[15] = state[11];
    state[11] = state[7];
    state[7] = temp;
}




__device__ void InvShiftRows(unsigned char* state) {
    unsigned char temp;

    temp = state[13];
    state[13] = state[9];
    state[9] = state[5];
    state[5] = state[1];
    state[1] = temp;


    temp = state[2];
    state[2] = state[10];
    state[10] = temp;
    temp = state[6];
    state[6] = state[14];
    state[14] = temp;


    temp = state[3];
    state[3] = state[7];
    state[7] = state[11];
    state[11] = state[15];
    state[15] = temp;
}


__device__ unsigned char gmul(unsigned char a, unsigned char b) {

    unsigned char p = 0;
    unsigned char high_bit_mask = 0x80;
    unsigned char high_bit;
    unsigned char modulo = 0x1B; /* x^8 + x^4 + x^3 + x + 1 */

    for (int i = 0; i < 8; i++) {
        if (b & 1) {
            p ^= a;
        }
        high_bit = a & high_bit_mask;
        a <<= 1;
        if (high_bit) {
            a ^= modulo;
        }
        b >>= 1;
    }
    return p;
}


__global__ void precompute_gmul_table_aes128() {
    for (int a = 0; a < 256; ++a) {
        for (int b = 0; b < 256; ++b) {
            gmul_table[a][b] = gmul(a, b);
        }
    }
}



__device__ void MixColumns(unsigned char* state) {
    unsigned char tmp[16];

    for (int i = 0; i < 4; i++) {
        tmp[i * 4 + 0] = gmul_table[0x02][state[i * 4 + 0]] ^ gmul_table[0x03][state[i * 4 + 1]] ^ state[i * 4 + 2] ^ state[i * 4 + 3];
        tmp[i * 4 + 1] = state[i * 4 + 0] ^ gmul_table[0x02][state[i * 4 + 1]] ^ gmul_table[0x03][state[i * 4 + 2]] ^ state[i * 4 + 3];
        tmp[i * 4 + 2] = state[i * 4 + 0] ^ state[i * 4 + 1] ^ gmul_table[0x02][state[i * 4 + 2]] ^ gmul_table[0x03][state[i * 4 + 3]];
        tmp[i * 4 + 3] = gmul_table[0x03][state[i * 4 + 0]] ^ state[i * 4 + 1] ^ state[i * 4 + 2] ^ gmul_table[0x02][state[i * 4 + 3]];
    }

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}





__device__ void InvMixColumns(unsigned char* state) {
    unsigned char tmp[16];

    for (int i = 0; i < 4; i++) {
        tmp[i * 4 + 0] = gmul_table[0x0e][state[i * 4 + 0]] ^ gmul_table[0x0b][state[i * 4 + 1]] ^ gmul_table[0x0d][state[i * 4 + 2]] ^ gmul_table[0x09][state[i * 4 + 3]];
        tmp[i * 4 + 1] = gmul_table[0x09][state[i * 4 + 0]] ^ gmul_table[0x0e][state[i * 4 + 1]] ^ gmul_table[0x0b][state[i * 4 + 2]] ^ gmul_table[0x0d][state[i * 4 + 3]];
        tmp[i * 4 + 2] = gmul_table[0x0d][state[i * 4 + 0]] ^ gmul_table[0x09][state[i * 4 + 1]] ^ gmul_table[0x0e][state[i * 4 + 2]] ^ gmul_table[0x0b][state[i * 4 + 3]];
        tmp[i * 4 + 3] = gmul_table[0x0b][state[i * 4 + 0]] ^ gmul_table[0x0d][state[i * 4 + 1]] ^ gmul_table[0x09][state[i * 4 + 2]] ^ gmul_table[0x0e][state[i * 4 + 3]];
    }

    for (int i = 0; i < 16; i++) {
        state[i] = tmp[i];
    }
}





__device__ void AddRoundKey(unsigned char* state, const unsigned char* roundKey) {
    for (int i = 0; i < 16; i++) {
        state[i] ^= roundKey[i];
    }
}




__global__ void initialize_start_key(unsigned char* startKey, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx == 0) { 
        curandState state;
        curand_init(seed, 0, 0, &state);
        for (int i = 0; i < 16; ++i) {
            startKey[i] = curand(&state) % 256;
        }
    }
}









__global__ void generate_sequential_aes_keys(unsigned char* keys, int n, unsigned char* startKey) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        unsigned char carry = 0;
        for (int i = 15; i >= 0; --i) {
            unsigned int keyOffset = (unsigned int)startKey[i] + carry;
            
            if (i == 15) keyOffset += idx;
            keys[idx * 16 + i] = keyOffset % 256;
            carry = keyOffset / 256;
        }

        
        int i = 14;
        while (carry > 0 && i >= 0) {
            unsigned int keyOffset = (unsigned int)keys[idx * 16 + i] + carry;
            keys[idx * 16 + i] = keyOffset % 256;
            carry = keyOffset / 256;
            --i;
        }
    }
}




void increment_key(unsigned char* key, unsigned int n) {
    unsigned int carry = n;
    for (int i = 15; i >= 0 && carry > 0; --i) {
        unsigned int current = key[i] + carry;
        key[i] = current % 256;
        carry = current / 256;
    }
}





__global__ void generate_aes_keys(unsigned char* keys, int n, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);

        for (int i = 0; i < 16; ++i) {
            keys[idx * 16 + i] = curand(&state) % 256;
        }
    }
}



__device__ void AESEncrypt(const unsigned char* plainText, unsigned char* key, unsigned char* cipherText) {
    unsigned char roundKeys[176];
    KeyExpansion(key, roundKeys);

    
    for (int i = 0; i < 16; ++i) {
        cipherText[i] = plainText[i];
    }

    
    AddRoundKey(cipherText, roundKeys);

    
    for (int round = 1; round < 10; ++round) {
        SubBytes(cipherText);
        ShiftRows(cipherText);
        MixColumns(cipherText);
        AddRoundKey(cipherText, roundKeys + round * 16);
    }

    
    SubBytes(cipherText);
    ShiftRows(cipherText);
    AddRoundKey(cipherText, roundKeys + 160);
}




__device__ void AESDecrypt(const unsigned char* cipherText, unsigned char* key, unsigned char* decryptedText) {
    unsigned char roundKeys[176];
    KeyExpansion(key, roundKeys);


    for (int i = 0; i < 16; ++i) {
        decryptedText[i] = cipherText[i];
    }

    AddRoundKey(decryptedText, roundKeys + 160);


    for (int round = 9; round > 0; --round) {
        InvShiftRows(decryptedText);
        InvSubBytes(decryptedText);
        AddRoundKey(decryptedText, roundKeys + round * 16);
        InvMixColumns(decryptedText);
    }


    InvShiftRows(decryptedText);
    InvSubBytes(decryptedText);
    AddRoundKey(decryptedText, roundKeys);
}


__global__ void tryDecryptAES(const unsigned char* cipherText, const unsigned char* expectedPlainText, unsigned char* keys, bool* found, unsigned char* foundKey, unsigned char* decryptedTexts) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned char decryptedText[16];

    AESDecrypt(cipherText, keys + idx * 16, decryptedText);

    

#if DEBUG_MODE

    for (int i = 0; i < 16; ++i) {
        decryptedTexts[idx * 16 + i] = decryptedText[i];
    }

#endif
   
    
    bool match = true;
    for (int i = 0; i < 16; ++i) {
        if (decryptedText[i] != expectedPlainText[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        *found = true;
        for (int i = 0; i < 16; ++i) {
            foundKey[i] = keys[idx * 16 + i];
        }
    }
   
    
}






int main(int argc, char* argv[]) {

    const int n = 116736; // Total de threads    
    int threadsPerBlock = 1024;
    int numberOfBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    unsigned long long seed = (unsigned long long)time(NULL);

    unsigned char* d_keys, * d_cipherText, * d_expectedPlainText, * d_foundKey, foundKey[16];
    bool found = false, * d_found;

    unsigned char* d_startKey;

    unsigned char* d_decryptedTexts;
    size_t decryptedTextsSize = n * 16 * sizeof(unsigned char); 
    cudaMalloc(&d_decryptedTexts, decryptedTextsSize);




#if DEBUG_MODE
    unsigned char* h_keys = (unsigned char*)malloc(n * 16 * sizeof(unsigned char));
#endif




    //  cipherText and expectedPlainText 
    const unsigned char cipherText[16] = { 0x8B, 0x66, 0x68, 0xC2, 0x7D, 0x22, 0x61, 0x05, 0xA9, 0x17, 0xD6, 0x61, 0x41, 0xBC, 0x7B, 0x67 };
    const unsigned char expectedPlainText[16] = { 0xC4, 0x93, 0xE8, 0x4A, 0xAD, 0xD1, 0xC3, 0x03, 0x91, 0x3A, 0xBD, 0x57, 0xFE, 0x09, 0x79, 0x36 };

    precompute_gmul_table_aes128 << <1, 1 >> > ();
    cudaDeviceSynchronize();

    
    cudaMalloc(&d_keys, n * 16 * sizeof(unsigned char));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_foundKey, 16 * sizeof(unsigned char));
    cudaMalloc(&d_cipherText, 16 * sizeof(unsigned char));
    cudaMalloc(&d_expectedPlainText, 16 * sizeof(unsigned char));
    cudaMalloc(&d_startKey, 16 * sizeof(unsigned char));



    
    cudaMemcpy(d_found, &found, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cipherText, cipherText, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_expectedPlainText, expectedPlainText, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);

   

    initialize_start_key << <1, 1 >> > (d_startKey, seed);
    cudaDeviceSynchronize();



    unsigned char h_startKey[16];
    cudaMemcpy(h_startKey, d_startKey, 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);



    unsigned char startKey[16] = { 0 }; 
    memcpy(startKey, h_startKey, 16); 


    printf("StartKey: ");
    for (int i = 0; i < 16; ++i) {
        printf("%02x ", h_startKey[i]); 
    }
    printf("\n");



#if TIMER
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#endif

    

    while (!found)
    {

#if TIMER
        cudaEventRecord(start);
#endif


        seed++;
        //generate_aes_keys << <numberOfBlocks, threadsPerBlock >> > (d_keys, n, seed);

        cudaMemcpy(d_startKey, startKey, 16 * sizeof(unsigned char), cudaMemcpyHostToDevice);
        generate_sequential_aes_keys << <numberOfBlocks, threadsPerBlock >> > (d_keys, n, d_startKey);
        cudaDeviceSynchronize();
        
        tryDecryptAES << <numberOfBlocks, threadsPerBlock >> > (d_cipherText, d_expectedPlainText, d_keys, d_found, d_foundKey, d_decryptedTexts);
        cudaDeviceSynchronize();
        

        cudaMemcpy(&found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);  


        increment_key(startKey, n);




#if TIMER
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution time for %d threads : %.2f milliseconds\n", n, milliseconds);

#endif






    

#if DEBUG_MODE

        unsigned char* h_decryptedTexts = (unsigned char*)malloc(n * 16 * sizeof(unsigned char));
        cudaMemcpy(h_decryptedTexts, d_decryptedTexts, n * 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_keys, d_keys, n * 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        

        for (int i = 0; i < n; ++i) {
            // Affichage du Cipher constant
            printf("CIPHER:    ");
            for (int j = 0; j < 16; ++j) {
                printf("%02x ", cipherText[j]);  // Supposons que cipherText est disponible ici
            }
            printf("\n");

            // Affichage de la Clé Générée
            printf("KEY:       ");
            for (int j = 0; j < 16; ++j) {
                printf("%02x ", h_keys[i * 16 + j]);
            }
            printf("\n");

            // Affichage du Texte Déchiffré
            printf("DECRYPTED: ");
            for (int j = 0; j < 16; ++j) {
                printf("%02x ", h_decryptedTexts[i * 16 + j]);
            }
            printf("\n\n");
    }

        free(h_decryptedTexts);


#endif



          
    }
    

    if (found) 
	{
        cudaMemcpy(foundKey, d_foundKey, 16 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        // Affichage de la clé 
        printf("Key Found !!! : ");
        for (int i = 0; i < 16; ++i) {
            printf("%02x ", foundKey[i]);
        }
        printf("\n");
    }



    

#if DEBUG_MODE
    if (h_keys != NULL) free(h_keys);
#endif 


    cudaFree(d_startKey);
    cudaFree(d_keys);
    cudaFree(d_found);
    cudaFree(d_foundKey);
    cudaFree(d_cipherText);
    cudaFree(d_expectedPlainText);
    cudaFree(d_decryptedTexts);

    return 0;
}




