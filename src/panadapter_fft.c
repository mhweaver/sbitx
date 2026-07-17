#include "panadapter_fft.h"
#include "sdr.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MIN_ANALYSIS_BANDWIDTH 5000
#define FILTER_TAPS 127
/*
 * Post-decimation sample rate as a percentage of displayed bandwidth.
 * 250 means 2.5x; raise it for more anti-alias guard band or lower it for
 * finer FFT resolution.
 */
#define DECIMATION_GUARD_PERCENT 250
#define MAX_DECIMATION ((100 * SDR_SAMPLE_RATE) / \
	(DECIMATION_GUARD_PERCENT * MIN_ANALYSIS_BANDWIDTH))
#define MAX_RAW_SAMPLES ((PANADAPTER_FFT_MAX_BINS - 1) * MAX_DECIMATION + FILTER_TAPS)
#define RING_SIZE (PANADAPTER_FFT_MAX_BINS * (MAX_DECIMATION + 1))
#define RING_INDEX_MASK (RING_SIZE - 1)
/* Blend 30% of each new non-CW spectrum into the displayed frame. */
#define NON_CW_NEW_FRAME_WEIGHT 0.3f

_Static_assert(RING_SIZE >= MAX_RAW_SAMPLES, "panadapter FFT ring must hold the largest analysis window");
_Static_assert(RING_SIZE > 0 && (RING_SIZE & (RING_SIZE - 1)) == 0, "panadapter FFT ring size must be a power of two");

struct panadapter_fft {
	fftwf_complex sample_ring[RING_SIZE];
	_Atomic uint64_t samples_written;

	fftwf_complex raw_work[RING_SIZE];
	fftwf_complex *fft_data;
	fftwf_plan fft_plan;
	int planned_fft_bins;
	float filter_coeff[FILTER_TAPS];
	int filter_bandwidth_hz;
	float smoothed_bins[PANADAPTER_FFT_FRAME_BINS];
	struct panadapter_fft_config smoothed_config;
	bool smoothed_config_valid;

	pthread_t worker_thread;
	pthread_mutex_t state_mutex;
	pthread_cond_t request_cond;
	struct panadapter_fft_config pending_config;
	uint64_t pending_serial;
	uint64_t last_request_ms;
	bool request_pending;
	bool last_request_valid;
	bool stop_worker;
	bool reset_smoothing;
	struct panadapter_fft_frame published_frame;
};

/** Return whether two configurations produce equivalent spectrum frames. */
static bool fft_configs_equal(const struct panadapter_fft_config *a,
							  const struct panadapter_fft_config *b)
{
	return a->display_span_hz == b->display_span_hz
	       && a->center_hz == b->center_hz
	       && a->is_cw == b->is_cw
	       && a->is_tx == b->is_tx
	       && a->fft_bins == b->fft_bins
	       && (!a->is_cw || a->wpm == b->wpm);

}

static uint64_t monotonic_ms(void)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t)now.tv_sec * 1000 + now.tv_nsec / 1000000;
}

/** Return the sample count to observe, capped at one dit for CW. */
static int observation_samples(const struct panadapter_fft_config *config,
							   int decimation)
{
	if (!config->is_cw)
		return config->fft_bins;

	const int wpm = config->wpm > 0 ? config->wpm : 1;
	const double sample_rate = (double)SDR_SAMPLE_RATE / decimation;
	int count = (int)lround(sample_rate * 1.2 / wpm);
	if (count < 1) count = 1;
	if (count > config->fft_bins) count = config->fft_bins;
	return count;
}

/** Reuse or create the in-place FFTW buffer and plan for the requested size. */
static bool ensure_fft_plan(struct panadapter_fft *state,
							int fft_bins)
{
	if (fft_bins == state->planned_fft_bins)
		return true;

	fftwf_complex *const new_data = fftwf_alloc_complex(fft_bins);
	if (!new_data)
		return false;
	fftwf_plan new_plan = fftwf_plan_dft_1d(fft_bins, new_data, new_data,
										FFTW_FORWARD, FFTW_ESTIMATE);
	if (!new_plan) {
		fftwf_free(new_data);
		return false;
	}

	if (state->fft_plan) fftwf_destroy_plan(state->fft_plan);
	if (state->fft_data) fftwf_free(state->fft_data);
	state->fft_data = new_data;
	state->fft_plan = new_plan;
	state->planned_fft_bins = fft_bins;
	return true;
}

/** Design a normalized Hamming-windowed sinc low-pass filter. */
static void make_filter(struct panadapter_fft *state,
						int bandwidth_hz)
{
	const int middle = FILTER_TAPS / 2;
	const double cutoff = 0.55 * bandwidth_hz / SDR_SAMPLE_RATE;
	double sum = 0.0;

	for (int tap = 0; tap < FILTER_TAPS; tap++) {
		const int offset = tap - middle;
		const double sinc = offset == 0 ? 2.0 * cutoff
			: sin(2.0 * M_PI * cutoff * offset) / (M_PI * offset);
		const double window = 0.54 - 0.46 * cos(2.0 * M_PI * tap /
											 (FILTER_TAPS - 1));
		state->filter_coeff[tap] = (float)(sinc * window);
		sum += state->filter_coeff[tap];
	}

	for (int tap = 0; tap < FILTER_TAPS; tap++)
		state->filter_coeff[tap] /= (float)sum;
}

/** Copy the newest ring samples into the work buffer without blocking audio. */
static bool snapshot_samples(struct panadapter_fft *state,
							 int count, uint64_t *first_sample,
							 uint64_t *sample_end_ms)
{
	const uint64_t end = atomic_load_explicit(&state->samples_written,
		memory_order_acquire);
	if (end < (uint64_t)count)
		return false;
	*sample_end_ms = monotonic_ms();

	const uint64_t start = end - (uint64_t)count;
	const size_t sample_count = (size_t)count;
	const size_t ring_index = (size_t)(start & RING_INDEX_MASK);
	size_t first_count = RING_SIZE - ring_index;
	if (first_count > sample_count) first_count = sample_count;
	memcpy(state->raw_work, state->sample_ring + ring_index,
		   first_count * sizeof(state->raw_work[0]));
	memcpy(state->raw_work + first_count, state->sample_ring,
		   (sample_count - first_count) * sizeof(state->raw_work[0]));

	/* The spare half of the ring lets the worker snapshot without blocking audio. */
	if (atomic_load_explicit(&state->samples_written, memory_order_acquire) -
		start > RING_SIZE)
		return false;
	*first_sample = start;
	return true;
}

/** Mix the requested analysis center down to DC in place. */
static void shift_samples(struct panadapter_fft *state, int count,
					  int center_hz, uint64_t first_sample)
{
	if (center_hz == 0) {
		return;
	}

	const double step = -2.0 * M_PI * center_hz / SDR_SAMPLE_RATE;
	const double phase = fmod(step * (double)(first_sample % SDR_SAMPLE_RATE),
		2.0 * M_PI);
	float oscillator_i = cosf((float)phase);
	float oscillator_q = sinf((float)phase);
	const float step_i = cosf((float)step);
	const float step_q = sinf((float)step);

	for (int index = 0; index < count; index++) {
		state->raw_work[index] *= oscillator_i + I * oscillator_q;

		const float next_i = oscillator_i * step_i - oscillator_q * step_q;
		oscillator_q = oscillator_i * step_q + oscillator_q * step_i;
		oscillator_i = next_i;
		if ((index & 1023) == 1023) {
			const float magnitude = hypotf(oscillator_i, oscillator_q);
			oscillator_i /= magnitude;
			oscillator_q /= magnitude;
		}
	}
}

/** Produce one cropped, smoothed display spectrum from the newest samples. */
static bool analyze(struct panadapter_fft *state,
					const struct panadapter_fft_config *config,
					struct panadapter_fft_frame *frame)
{
	const int fft_bins = config->fft_bins;
	if (fft_bins < 1 || fft_bins > PANADAPTER_FFT_MAX_BINS ||
		(fft_bins & (fft_bins - 1)) != 0 ||
		!ensure_fft_plan(state, fft_bins))
		return false;

	const int bandwidth = config->display_span_hz > MIN_ANALYSIS_BANDWIDTH
		? config->display_span_hz : MIN_ANALYSIS_BANDWIDTH;
	const int guarded_decimation = (100 * SDR_SAMPLE_RATE) /
		(DECIMATION_GUARD_PERCENT * bandwidth);
	const int decimation = guarded_decimation > 0 ? guarded_decimation : 1;
	const int observed = observation_samples(config, decimation);
	const int raw_count = (observed - 1) * decimation + FILTER_TAPS;
	uint64_t first_sample;
	uint64_t sample_end_ms;

	if (!snapshot_samples(state, raw_count, &first_sample, &sample_end_ms))
		return false;

	if (bandwidth != state->filter_bandwidth_hz) {
		make_filter(state, bandwidth);
		state->filter_bandwidth_hz = bandwidth;
	}
	shift_samples(state, raw_count, config->center_hz, first_sample);
	memset(state->fft_data, 0, sizeof(*state->fft_data) * fft_bins);

	// Keep signal levels independent of FFT length, preserving 2048-bin levels.
	const float length_scale = (float)PANADAPTER_FFT_DEFAULT_BINS / (float)observed;
	for (int output = 0; output < observed; output++) {
		const int newest = FILTER_TAPS - 1 + output * decimation;
		fftwf_complex sum = 0.0f;
		for (int tap = 0; tap < FILTER_TAPS; tap++) {
			const int input = newest - tap;
			sum += state->filter_coeff[tap] * state->raw_work[input];
		}
		const float window = observed == 1 ? 1.0f
			: 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)output /
										(float)(observed - 1));
		state->fft_data[output] = sum * window * length_scale;
	}

	fftwf_execute(state->fft_plan);

	if (!state->smoothed_config_valid ||
		!fft_configs_equal(config, &state->smoothed_config)) {
		memset(state->smoothed_bins, 0, sizeof(state->smoothed_bins));
		state->smoothed_config = *config;
		state->smoothed_config_valid = true;
	}

	const double output_rate = (double)SDR_SAMPLE_RATE / decimation;
	const double bin_hz = output_rate / fft_bins;
	int half_bins = (int)floor(config->display_span_hz / (2.0 * bin_hz));
	const int max_half_bins = fft_bins > 1 ? fft_bins / 2 - 1 : 0;
	if (half_bins > max_half_bins)
		half_bins = max_half_bins;
	const int visible_bins = 2 * half_bins + 1;
	frame->count = visible_bins < PANADAPTER_FFT_FRAME_BINS
		? visible_bins : PANADAPTER_FFT_FRAME_BINS;
	frame->first_hz = config->center_hz + half_bins * bin_hz;
	frame->bin_step_hz = frame->count > 1
		? -(visible_bins - 1) * bin_hz / (frame->count - 1) : -bin_hz;
	frame->decimation = decimation;
	frame->observation_samples = observed;
	frame->sample_end_ms = sample_end_ms;
	frame->config = *config;

	// CW timing belongs in the waterfall rows, not a multi-frame magnitude tail.
	const float new_frame_weight = config->is_cw
		? 1.0f : NON_CW_NEW_FRAME_WEIGHT;
	for (int output = 0; output < frame->count; output++) {
		const int first_visible = output * visible_bins / frame->count;
		const int end_visible = (output + 1) * visible_bins / frame->count;
		float magnitude = 0.0f;
		for (int visible = first_visible; visible < end_visible; visible++) {
			const int signed_bin = half_bins - visible;
			const int fft_bin = signed_bin >= 0 ? signed_bin : fft_bins + signed_bin;
			magnitude = fmaxf(magnitude, cabsf(state->fft_data[fft_bin]));
		}
		state->smoothed_bins[output] =
			(1.0f - new_frame_weight) * state->smoothed_bins[output] +
			new_frame_weight * magnitude;
		frame->bins[output] = (int)lroundf(20.0f *
								 log10f(fmaxf(state->smoothed_bins[output],
											  1.0e-12f)));
	}
	return true;
}

/** Process coalesced requests and publish only the newest analysis result. */
static void *panadapter_fft_worker(void *context)
{
	struct panadapter_fft *const state = context;
	while (true) {
		pthread_mutex_lock(&state->state_mutex);
		while (!state->request_pending && !state->stop_worker)
			pthread_cond_wait(&state->request_cond, &state->state_mutex);
		if (state->stop_worker) {
			pthread_mutex_unlock(&state->state_mutex);
			break;
		}
		const struct panadapter_fft_config config = state->pending_config;
		const uint64_t serial = state->pending_serial;
		const bool reset = state->reset_smoothing;
		state->reset_smoothing = false;
		state->request_pending = false;
		pthread_mutex_unlock(&state->state_mutex);
		if (reset)
			state->smoothed_config_valid = false;

		struct panadapter_fft_frame frame;
		if (!analyze(state, &config, &frame))
			continue;

		pthread_mutex_lock(&state->state_mutex);
		if (serial == state->pending_serial) {
			frame.generation = serial;
			state->published_frame = frame;
		}
		pthread_mutex_unlock(&state->state_mutex);
	}
	if (state->fft_plan) fftwf_destroy_plan(state->fft_plan);
	if (state->fft_data) fftwf_free(state->fft_data);
	return NULL;
}

struct panadapter_fft *panadapter_fft_create(void)
{
	struct panadapter_fft *const state = calloc(1, sizeof(*state));
	if (!state)
		return NULL;

	atomic_init(&state->samples_written, 0);
	if (pthread_mutex_init(&state->state_mutex, NULL) != 0) {
		free(state);
		return NULL;
	}
	if (pthread_cond_init(&state->request_cond, NULL) != 0) {
		pthread_mutex_destroy(&state->state_mutex);
		free(state);
		return NULL;
	}
	if (pthread_create(&state->worker_thread, NULL,
		panadapter_fft_worker, state) != 0) {
		pthread_cond_destroy(&state->request_cond);
		pthread_mutex_destroy(&state->state_mutex);
		free(state);
		return NULL;
	}
	return state;
}

void panadapter_fft_destroy(struct panadapter_fft *state)
{
	if (!state)
		return;

	pthread_mutex_lock(&state->state_mutex);
	state->stop_worker = true;
	pthread_cond_signal(&state->request_cond);
	pthread_mutex_unlock(&state->state_mutex);
	pthread_join(state->worker_thread, NULL);
	pthread_cond_destroy(&state->request_cond);
	pthread_mutex_destroy(&state->state_mutex);
	free(state);
}

/** Append complex input samples to the lock-free analysis ring. */
void panadapter_fft_push(struct panadapter_fft *state,
	const double *i_samples, const double *q_samples, int count)
{
	if (!state)
		return;

	const uint64_t start = atomic_load_explicit(&state->samples_written,
		memory_order_relaxed);
	for (int index = 0; index < count; index++) {
		const uint64_t ring_index = (start + index) & RING_INDEX_MASK;
		state->sample_ring[ring_index] =
			(float)i_samples[index] + I * (float)q_samples[index];
	}
	atomic_store_explicit(&state->samples_written, start + count,
		memory_order_release);
}

/** Queue the newest analysis request, subject to the configured refresh rate. */
void panadapter_fft_request(struct panadapter_fft *state,
	const struct panadapter_fft_config *config)
{
	if (!state || !config)
		return;

	const uint64_t now = monotonic_ms();
	pthread_mutex_lock(&state->state_mutex);
	if (state->last_request_valid &&
		fft_configs_equal(config, &state->pending_config) &&
		now - state->last_request_ms < (uint64_t)(config->refresh_ms > 0
										 ? config->refresh_ms : 0)) {
		pthread_mutex_unlock(&state->state_mutex);
		return;
	}
	state->pending_config = *config;
	state->last_request_valid = true;
	state->last_request_ms = now;
	state->pending_serial++;
	state->request_pending = true;
	pthread_cond_signal(&state->request_cond);
	pthread_mutex_unlock(&state->state_mutex);
}

/** Non-blockingly copy the latest frame matching the requested analysis. */
bool panadapter_fft_get_frame(struct panadapter_fft *state,
	const struct panadapter_fft_config *config,
	struct panadapter_fft_frame *frame)
{
	if (!state || !config || !frame ||
		pthread_mutex_trylock(&state->state_mutex) != 0)
		return false;
	const bool available = state->published_frame.generation != 0 &&
		fft_configs_equal(config, &state->published_frame.config);
	if (available)
		*frame = state->published_frame;
	pthread_mutex_unlock(&state->state_mutex);
	return available;
}

/** Estimate the age of the frame's effective observation center. */
int panadapter_fft_frame_latency_ms(const struct panadapter_fft_frame *frame)
{
	if (!frame || !frame->sample_end_ms || frame->observation_samples < 1 ||
		frame->decimation < 1)
		return -1;

	const uint64_t now = monotonic_ms();
	const uint64_t age_ms = now > frame->sample_end_ms
		? now - frame->sample_end_ms : 0;
	const double center_samples = (frame->observation_samples - 1) *
		frame->decimation / 2.0 + (FILTER_TAPS - 1) / 2.0;
	return (int)lround((double)age_ms +
		center_samples * 1000.0 / SDR_SAMPLE_RATE);
}

/** Discard pending and published frames and reset display smoothing. */
void panadapter_fft_reset(struct panadapter_fft *state)
{
	if (!state)
		return;
	pthread_mutex_lock(&state->state_mutex);
	state->reset_smoothing = true;
	state->pending_serial++;
	state->request_pending = false;
	state->last_request_valid = false;
	state->published_frame.generation = 0;
	pthread_mutex_unlock(&state->state_mutex);
}
