#ifndef ZOOM_FFT_H
#define ZOOM_FFT_H

#include <stdbool.h>
#include <stdint.h>

#define ZOOM_FFT_DEFAULT_BINS 2048
#define ZOOM_FFT_MAX_BINS 16384
#define ZOOM_FFT_FRAME_BINS 2048

struct zoom_fft_config {
	int display_span_hz;
	int center_hz;
	int is_cw;
	int is_tx;
	int wpm;
	int refresh_ms;
	int fft_bins;
};

struct zoom_fft_frame {
	int bins[ZOOM_FFT_FRAME_BINS];
	int count;
	double first_hz;
	double bin_step_hz;
	int analysis_bandwidth_hz;
	int decimation;
	int observation_samples;
	uint64_t generation;
	struct zoom_fft_config config;
};

int zoom_fft_init(void);
void zoom_fft_shutdown(void);
void zoom_fft_push(const double *i_samples, const double *q_samples, int count);
void zoom_fft_request(const struct zoom_fft_config *config);
bool zoom_fft_get_frame(const struct zoom_fft_config *config,
						  struct zoom_fft_frame *frame);
void zoom_fft_reset(void);

#endif
