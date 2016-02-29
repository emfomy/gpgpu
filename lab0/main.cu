#include <cstdio>
#include <cstdlib>
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

/// Define default number of blocks and threads
const auto kNumBlock = 4;
const auto kNumThread = 64;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Transform cases: convert lower cases to upper ones, and vice versa
///
__global__ void TransformCases( char *input_gpu, int fsize ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	auto &c = input_gpu[idx];
	if ( idx < fsize ) {
		if ( c >= 'a' && c <= 'z' ) {
			c += 'A'-'a';
		} else if ( c >= 'A' && c <= 'Z' ) {
			c += 'a'-'A';
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Caesar cipher: encrypt characters using Caesar cipher
///
/// output = input * key % 95
/// Will only convert the visible characters. The default key is 16.
///
__global__ void CaesarCipher( char *input_gpu, int fsize, int key = 16 ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	auto &c = input_gpu[idx];
	if ( idx < fsize && c >= 32 && c <= 126 ) {
		c = (static_cast<int>(c-32) * key) % 95 + 32;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Swap the characters: maps the vowels to vowels and consonant to consonant
///
/// Uses Caesar cipher: output = (input * 25 + 8) % 26
///
__global__ void SwapVowelConsonant( char *input_gpu, int fsize ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	auto &c = input_gpu[idx];
	if ( idx < fsize ) {
		if ( c >= 'a' && c <= 'z' ) {
			c = (static_cast<int>(c-'a') * 25 + 8) % 26 + 'a';
		} else if ( c >= 'A' && c <= 'Z' ) {
			c = (static_cast<int>(c-'A') * 25 + 8) % 26 + 'A';
		}
	}
}

int main(int argc, char **argv)
{
	// init, and check
	if (argc != 2) {
		printf("Usage %s <input text file>\n", argv[0]);
		abort();
	}
	FILE *fp = fopen(argv[1], "r");
	if (not fp) {
		printf("Cannot open %s", argv[1]);
		abort();
	}
	// get file size
	fseek(fp, 0, SEEK_END);
	size_t fsize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	// read files
	MemoryBuffer<char> text(fsize+1);
	auto text_smem = text.CreateSync(fsize);
	CHECK;
	fread(text_smem.get_cpu_wo(), 1, fsize, fp);
	text_smem.get_cpu_wo()[fsize] = '\0';
	fclose(fp);

	// TODO: do your transform here
	char *input_gpu = text_smem.get_gpu_rw();
	// An example: transform the first 64 characters to '!'
	// Don't transform over the tail
	// And don't transform the line breaks
	while ( static_cast<long>(fsize) > 0 ) {
		SwapVowelConsonant<<<kNumBlock, kNumThread>>>(input_gpu, fsize);
		input_gpu += kNumBlock * kNumThread;
		fsize     -= kNumBlock * kNumThread;
	}

	puts(text_smem.get_cpu_ro());
	return 0;
}
