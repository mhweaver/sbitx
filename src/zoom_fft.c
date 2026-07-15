#include "zoom_fft.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_RATE 96000.0
#define MIN_ANALYSIS_BANDWIDTH 5000
#define RING_SIZE 32768
#define RING_MASK (RING_SIZE - 1)
#define FILTER_TAPS 127
#define ZOOM_SMOOTHING_SPEED 0.3f

static fftwf_complex sample_ring[RING_SIZE];
static fftwf_complex raw_work[RING_SIZE];
static fftwf_complex shifted_work[RING_SIZE];
static _Atomic uint64_t samples_written;

static fftwf_complex *fft_input;
static fftwf_complex *fft_output;
static fftwf_plan fft_plan;
static float filter_coeff[FILTER_TAPS];
static int filter_bandwidth_hz;
static float smoothed_bins[ZOOM_FFT_BINS];
static struct zoom_fft_config smoothed_config;
static bool smoothed_config_valid;

static pthread_t worker_thread;
static pthread_mutex_t request_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t request_cond = PTHREAD_COND_INITIALIZER;
static struct zoom_fft_config pending_config;
static struct zoom_fft_config last_requested_config;
static uint64_t pending_serial;
static uint64_t last_request_ms;
static bool request_pending;
static bool last_request_valid;
static bool stop_worker;

static pthread_mutex_t frame_mutex = PTHREAD_MUTEX_INITIALIZER;
static struct zoom_fft_frame published_frame;
static bool frame_ready;

static _Atomic bool initialized;
static _Atomic bool display_active;
static _Atomic bool reset_smoothing;

static bool same_analysis(const struct zoom_fft_config *a,
						  const struct zoom_fft_config *b)
{
	return a->display_span_hz == b->display_span_hz &&
		   a->center_hz == b->center_hz && a->is_cw == b->is_cw &&
		   (!a->is_cw || a->wpm == b->wpm);
}

static uint64_t monotonic_ms(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

static int analysis_bandwidth(const struct zoom_fft_config *config)
{
	return config->display_span_hz > MIN_ANALYSIS_BANDWIDTH
		? config->display_span_hz : MIN_ANALYSIS_BANDWIDTH;
}

static int analysis_decimation(const struct zoom_fft_config *config)
{
	int decimation = (int)floor(INPUT_RATE / (2.5 * analysis_bandwidth(config)));
	return decimation > 0 ? decimation : 1;
}

static int observation_samples(const struct zoom_fft_config *config, int decimation)
{
	if (!config->is_cw)
		return ZOOM_FFT_BINS;

	int wpm = config->wpm > 0 ? config->wpm : 1;
	double sample_rate = INPUT_RATE / decimation;
	int count = (int)lround(sample_rate * 1.2 / wpm);
	if (count < 1) count = 1;
	if (count > ZOOM_FFT_BINS) count = ZOOM_FFT_BINS;
	return count;
}

bool zoom_fft_should_use(int display_span_hz, int plot_width)
{
	if (display_span_hz <= 0 || plot_width <= 0)
		return false;

	double legacy_bin_pixels = (INPUT_RATE / ZOOM_FFT_BINS) * plot_width /
								 display_span_hz;
	struct zoom_fft_config config = {.display_span_hz = display_span_hz};
	return legacy_bin_pixels > 2.0 && analysis_decimation(&config) > 1;
}

static void make_filter(int bandwidth_hz)
{
	const int middle = FILTER_TAPS / 2;
	const double cutoff = 0.55 * bandwidth_hz / INPUT_RATE;
	double sum = 0.0;

	for (int tap = 0; tap < FILTER_TAPS; tap++) {
		int offset = tap - middle;
		double sinc = offset == 0 ? 2.0 * cutoff
			: sin(2.0 * M_PI * cutoff * offset) / (M_PI * offset);
		double window = 0.54 - 0.46 * cos(2.0 * M_PI * tap /
											 (FILTER_TAPS - 1));
		filter_coeff[tap] = (float)(sinc * window);
		sum += filter_coeff[tap];
	}

	for (int tap = 0; tap < FILTER_TAPS; tap++)
		filter_coeff[tap] /= (float)sum;
}

static bool snapshot_samples(int count, uint64_t *first_sample)
{
	uint64_t end = atomic_load_explicit(&samples_written, memory_order_acquire);
	if (end < (uint64_t)count)
		return false;

	uint64_t start = end - count;
	for (int index = 0; index < count; index++) {
		uint64_t ring_index = (start + index) & RING_MASK;
		raw_work[index] = sample_ring[ring_index];
	}

	/* The spare half of the ring lets the worker snapshot without blocking audio. */
	if (atomic_load_explicit(&samples_written, memory_order_acquire) - start > RING_SIZE)
		return false;
	*first_sample = start;
	return true;
}

static void shift_samples(int count, int center_hz, uint64_t first_sample)
{
	if (center_hz == 0) {
		memcpy(shifted_work, raw_work, count * sizeof(raw_work[0]));
		return;
	}

	double step = -2.0 * M_PI * center_hz / INPUT_RATE;
	double phase = fmod(step * (double)(first_sample % 96000), 2.0 * M_PI);
	float oscillator_i = cosf((float)phase);
	float oscillator_q = sinf((float)phase);
	float step_i = cosf((float)step);
	float step_q = sinf((float)step);

	for (int index = 0; index < count; index++) {
		shifted_work[index] = raw_work[index] * (oscillator_i + I * oscillator_q);

		float next_i = oscillator_i * step_i - oscillator_q * step_q;
		oscillator_q = oscillator_i * step_q + oscillator_q * step_i;
		oscillator_i = next_i;
		if ((index & 1023) == 1023) {
			float magnitude = hypotf(oscillator_i, oscillator_q);
			oscillator_i /= magnitude;
			oscillator_q /= magnitude;
		}
	}
}

static bool analyze(const struct zoom_fft_config *config,
					struct zoom_fft_frame *frame)
{
	int bandwidth = analysis_bandwidth(config);
	int decimation = analysis_decimation(config);
	int observed = observation_samples(config, decimation);
	int raw_count = (observed - 1) * decimation + FILTER_TAPS;
	uint64_t first_sample;

	if (raw_count > RING_SIZE || !snapshot_samples(raw_count, &first_sample))
		return false;

	if (bandwidth != filter_bandwidth_hz) {
		make_filter(bandwidth);
		filter_bandwidth_hz = bandwidth;
	}
	shift_samples(raw_count, config->center_hz, first_sample);
	memset(fft_input, 0, sizeof(*fft_input) * ZOOM_FFT_BINS);

	float length_scale = (float)ZOOM_FFT_BINS / observed;
	for (int output = 0; output < observed; output++) {
		int newest = FILTER_TAPS - 1 + output * decimation;
		float sum_i = 0.0f;
		float sum_q = 0.0f;
		for (int tap = 0; tap < FILTER_TAPS; tap++) {
			int input = newest - tap;
			sum_i += filter_coeff[tap] * crealf(shifted_work[input]);
			sum_q += filter_coeff[tap] * cimagf(shifted_work[input]);
		}
		float window = observed == 1 ? 1.0f
			: 0.5f - 0.5f * cosf(2.0f * (float)M_PI * output /
										(observed - 1));
		fft_input[output] = (sum_i + I * sum_q) * window * length_scale;
	}

	fftwf_execute(fft_plan);

	if (atomic_exchange_explicit(&reset_smoothing, false, memory_order_acq_rel) ||
		!smoothed_config_valid || !same_analysis(config, &smoothed_config)) {
		memset(smoothed_bins, 0, sizeof(smoothed_bins));
		smoothed_config = *config;
		smoothed_config_valid = true;
	}

	double output_rate = INPUT_RATE / decimation;
	double bin_hz = output_rate / ZOOM_FFT_BINS;
	int half_bins = (int)floor(config->display_span_hz / (2.0 * bin_hz));
	if (half_bins > ZOOM_FFT_BINS / 2 - 1)
		half_bins = ZOOM_FFT_BINS / 2 - 1;
	frame->count = 2 * half_bins + 1;
	frame->first_hz = config->center_hz + half_bins * bin_hz;
	frame->bin_step_hz = -bin_hz;
	frame->analysis_bandwidth_hz = bandwidth;
	frame->decimation = decimation;
	frame->observation_samples = observed;
	frame->config = *config;

	// CW timing belongs in the waterfall rows, not a multi-frame magnitude tail.
	float speed = config->is_cw ? 1.0f : ZOOM_SMOOTHING_SPEED;
	for (int output = 0; output < frame->count; output++) {
		int signed_bin = half_bins - output;
		int fft_bin = signed_bin >= 0 ? signed_bin : ZOOM_FFT_BINS + signed_bin;
		float magnitude = cabsf(fft_output[fft_bin]);
		smoothed_bins[output] = (1.0f - speed) * smoothed_bins[output] +
								 speed * magnitude;
		frame->bins[output] = (int)lroundf(20.0f *
								 log10f(fmaxf(smoothed_bins[output], 1.0e-12f)));
	}
	return true;
}

static void *zoom_worker(void *unused)
{
	(void)unused;
	for (;;) {
		pthread_mutex_lock(&request_mutex);
		while (!request_pending && !stop_worker)
			pthread_cond_wait(&request_cond, &request_mutex);
		if (stop_worker) {
			pthread_mutex_unlock(&request_mutex);
			break;
		}
		struct zoom_fft_config config = pending_config;
		uint64_t serial = pending_serial;
		request_pending = false;
		pthread_mutex_unlock(&request_mutex);

		struct zoom_fft_frame frame;
		if (!analyze(&config, &frame))
			continue;

		pthread_mutex_lock(&request_mutex);
		if (serial == pending_serial) {
			frame.generation = serial;
			pthread_mutex_lock(&frame_mutex);
			published_frame = frame;
			frame_ready = true;
			pthread_mutex_unlock(&frame_mutex);
		}
		pthread_mutex_unlock(&request_mutex);
	}
	return NULL;
}

int zoom_fft_init(void)
{
	if (atomic_load_explicit(&initialized, memory_order_acquire))
		return 0;

	fft_input = fftwf_malloc(sizeof(*fft_input) * ZOOM_FFT_BINS);
	fft_output = fftwf_malloc(sizeof(*fft_output) * ZOOM_FFT_BINS);
	if (!fft_input || !fft_output)
		goto fail;
	fft_plan = fftwf_plan_dft_1d(ZOOM_FFT_BINS, fft_input, fft_output,
								 FFTW_FORWARD, FFTW_ESTIMATE);
	if (!fft_plan)
		goto fail;

	atomic_store_explicit(&samples_written, 0, memory_order_relaxed);
	stop_worker = false;
	request_pending = false;
	last_request_valid = false;
	frame_ready = false;
	smoothed_config_valid = false;
	filter_bandwidth_hz = 0;
	if (pthread_create(&worker_thread, NULL, zoom_worker, NULL) != 0)
		goto fail;
	atomic_store_explicit(&initialized, true, memory_order_release);
	return 0;

fail:
	if (fft_plan) fftwf_destroy_plan(fft_plan);
	if (fft_input) fftwf_free(fft_input);
	if (fft_output) fftwf_free(fft_output);
	fft_plan = NULL;
	fft_input = NULL;
	fft_output = NULL;
	return -1;
}

void zoom_fft_shutdown(void)
{
	if (!atomic_exchange_explicit(&initialized, false, memory_order_acq_rel))
		return;

	pthread_mutex_lock(&request_mutex);
	stop_worker = true;
	pthread_cond_signal(&request_cond);
	pthread_mutex_unlock(&request_mutex);
	pthread_join(worker_thread, NULL);
	fftwf_destroy_plan(fft_plan);
	fftwf_free(fft_input);
	fftwf_free(fft_output);
	fft_plan = NULL;
	fft_input = NULL;
	fft_output = NULL;
}

void zoom_fft_push(const double *i_samples, const double *q_samples, int count)
{
	if (!atomic_load_explicit(&initialized, memory_order_acquire))
		return;

	uint64_t start = atomic_load_explicit(&samples_written, memory_order_relaxed);
	for (int index = 0; index < count; index++) {
		uint64_t ring_index = (start + index) & RING_MASK;
		sample_ring[ring_index] = (float)i_samples[index] + I * (float)q_samples[index];
	}
	atomic_store_explicit(&samples_written, start + count, memory_order_release);
}

void zoom_fft_request(const struct zoom_fft_config *config)
{
	if (!atomic_load_explicit(&initialized, memory_order_acquire))
		return;

	uint64_t now = monotonic_ms();
	pthread_mutex_lock(&request_mutex);
	if (last_request_valid && same_analysis(config, &last_requested_config) &&
		now - last_request_ms < (uint64_t)(config->refresh_ms > 0
										 ? config->refresh_ms : 0)) {
		pthread_mutex_unlock(&request_mutex);
		return;
	}
	pending_config = *config;
	last_requested_config = *config;
	last_request_valid = true;
	last_request_ms = now;
	pending_serial++;
	request_pending = true;
	pthread_cond_signal(&request_cond);
	pthread_mutex_unlock(&request_mutex);
}

bool zoom_fft_get_frame(const struct zoom_fft_config *config,
						  struct zoom_fft_frame *frame)
{
	if (pthread_mutex_trylock(&frame_mutex) != 0)
		return false;
	bool available = frame_ready && same_analysis(config, &published_frame.config);
	if (available)
		*frame = published_frame;
	pthread_mutex_unlock(&frame_mutex);
	return available;
}

void zoom_fft_set_active(bool active)
{
	atomic_store_explicit(&display_active, active, memory_order_release);
}

bool zoom_fft_is_active(void)
{
	return atomic_load_explicit(&display_active, memory_order_acquire);
}

void zoom_fft_reset(void)
{
	atomic_store_explicit(&reset_smoothing, true, memory_order_release);
	zoom_fft_set_active(false);
	pthread_mutex_lock(&request_mutex);
	pending_serial++;
	request_pending = false;
	last_request_valid = false;
	pthread_mutex_unlock(&request_mutex);
	pthread_mutex_lock(&frame_mutex);
	frame_ready = false;
	pthread_mutex_unlock(&frame_mutex);
}
