#include "sdr.h"
#include "panadapter_fft.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TEST_RATE ((double)SDR_SAMPLE_RATE)
#define CHUNK 1024
#define TEST_RING_SIZE 131072

static void push_signal(struct panadapter_fft *context, int count,
						double frequency_hz, double amplitude)
{
	double i_samples[CHUNK];
	double q_samples[CHUNK];
	uint64_t sample_number = 0;
	while (count > 0) {
		const int block = count < CHUNK ? count : CHUNK;
		for (int index = 0; index < block; index++) {
			const double phase = 2.0 * M_PI * frequency_hz *
				sample_number++ / TEST_RATE;
			i_samples[index] = amplitude * cos(phase);
			q_samples[index] = amplitude * sin(phase);
		}
		panadapter_fft_push(context, i_samples, q_samples, block);
		count -= block;
	}
}

static void push_noise(struct panadapter_fft *context, int count,
					   double amplitude)
{
	double i_samples[CHUNK];
	double q_samples[CHUNK];
	uint32_t random = 1;
	while (count > 0) {
		const int block = count < CHUNK ? count : CHUNK;
		for (int index = 0; index < block; index++) {
			random = random * 1664525u + 1013904223u;
			i_samples[index] = amplitude *
				((double)(random >> 8) / 8388607.5 - 1.0);
			random = random * 1664525u + 1013904223u;
			q_samples[index] = amplitude *
				((double)(random >> 8) / 8388607.5 - 1.0);
		}
		panadapter_fft_push(context, i_samples, q_samples, block);
		count -= block;
	}
}

static struct panadapter_fft_frame wait_for_frame(
	struct panadapter_fft *context,
	const struct panadapter_fft_config *config, uint64_t after_generation)
{
	struct panadapter_fft_frame frame;
	panadapter_fft_request(context, config);
	for (int attempt = 0; attempt < 2000; attempt++) {
		if (panadapter_fft_get_frame(context, config, &frame) &&
			frame.generation > after_generation)
			return frame;
		usleep(1000);
	}
	assert(!"timed out waiting for panadapter FFT frame");
	return frame;
}

static struct panadapter_fft_frame *wait_for_history_batch(
	struct panadapter_fft *context,
	const struct panadapter_fft_config *config, const uint64_t *sample_ends,
	int count, uint64_t generation)
{
	assert(panadapter_fft_request_history_batch(context, config, sample_ends,
		count, generation));
	for (int attempt = 0; attempt < 2000; attempt++) {
		int result_count = 0;
		struct panadapter_fft_frame *frames =
			panadapter_fft_take_history_batch(context, generation, &result_count);
		if (frames) {
			assert(result_count == count);
			return frames;
		}
		usleep(1000);
	}
	assert(!"timed out waiting for historical panadapter FFT batch");
	return NULL;
}

static int peak_bin(const struct panadapter_fft_frame *frame)
{
	int peak = 0;
	for (int index = 1; index < frame->count; index++)
		if (frame->bins[index] > frame->bins[peak])
			peak = index;
	return peak;
}

static int peak_level(const struct panadapter_fft_frame *frame)
{
	return frame->bins[peak_bin(frame)];
}

static void assert_tone(struct panadapter_fft *context, double tone_hz,
						int span_hz, int center_hz,
						struct panadapter_fft_config *config, uint64_t *generation)
{
	push_signal(context, TEST_RING_SIZE, tone_hz, 0.25);
	config->display_span_hz = span_hz;
	config->center_hz = center_hz;
	config->is_cw = 0;
	const struct panadapter_fft_frame frame =
		wait_for_frame(context, config, *generation);
	*generation = frame.generation;
	assert(frame.config.display_width_px == config->display_width_px);
	assert(frame.count == config->display_width_px);
	const double peak_hz = frame.first_hz + peak_bin(&frame) * frame.bin_step_hz;
	assert(fabs(peak_hz - tone_hz) <= 1.5 * fabs(frame.bin_step_hz));
}

static void assert_display_width(struct panadapter_fft *context,
	int display_width_px, int expected_fft_bins,
	struct panadapter_fft_config *config, uint64_t *generation)
{
	push_signal(context, TEST_RING_SIZE, 0.0, 0.25);
	config->display_span_hz = 2500;
	config->center_hz = 0;
	config->is_cw = 0;
	config->display_width_px = display_width_px;
	const struct panadapter_fft_frame frame =
		wait_for_frame(context, config, *generation);
	*generation = frame.generation;
	assert(frame.fft_bins == expected_fft_bins);
	assert(frame.count == display_width_px);
	assert(frame.observation_samples == expected_fft_bins);
}

static int noise_level(struct panadapter_fft *context, int display_span_hz,
	int display_width_px, struct panadapter_fft_config *config,
	uint64_t *generation)
{
	push_noise(context, TEST_RING_SIZE, 0.1);
	config->display_span_hz = display_span_hz;
	config->center_hz = 0;
	config->is_cw = 0;
	config->display_width_px = display_width_px;
	const struct panadapter_fft_frame frame =
		wait_for_frame(context, config, *generation);
	*generation = frame.generation;
	int64_t sum = 0;
	for (int index = 0; index < frame.count; index++)
		sum += frame.bins[index];
	return (int)lround((double)sum / frame.count);
}

int main(void)
{
	struct panadapter_fft *const context = panadapter_fft_create();
	assert(context);

	struct panadapter_fft_config config = {
		.wpm = 20,
		.refresh_ms = 0,
		.display_width_px = 400,
	};
	uint64_t generation = 0;
	const int widths[] = {1, 200, 400, 800, 1600, 2048};
	const int expected_fft_bins[] = {1024, 2048, 4096, 8192, 16384, 16384};
	for (unsigned index = 0; index < sizeof(widths) / sizeof(widths[0]); index++) {
		assert_display_width(context, widths[index], expected_fft_bins[index],
			&config, &generation);
	}
	const int small_fft_noise = noise_level(context, 2500, 200,
		&config, &generation);
	const int large_fft_noise = noise_level(context, 2500, 1600,
		&config, &generation);
	assert(abs(large_fft_noise - small_fft_noise) <= 2);
	const int wide_span_noise = noise_level(context, 25000, 400,
		&config, &generation);
	const int narrow_span_noise = noise_level(context, 2500, 400,
		&config, &generation);
	assert(abs(narrow_span_noise - wide_span_noise) <= 4);

	config.display_width_px = 2048;
	assert_tone(context, 700.0, 2500, 0, &config, &generation);
	push_signal(context, TEST_RING_SIZE, 700.0, 0.25);
	const struct panadapter_fft_frame fine =
		wait_for_frame(context, &config, generation);
	generation = fine.generation;
	assert(fine.fft_bins == PANADAPTER_FFT_MAX_BINS);
	assert(fine.observation_samples == PANADAPTER_FFT_MAX_BINS);
	assert(fabs(fine.bin_step_hz) < 2.0);
	const int nominal_latency = (int)lround((fine.observation_samples - 1) *
		fine.decimation * 1000.0 / (2.0 * TEST_RATE));
	const int measured_latency = panadapter_fft_frame_latency_ms(&fine);
	assert(measured_latency >= nominal_latency);
	assert(measured_latency < nominal_latency + 1000);
	push_signal(context, TEST_RING_SIZE, -700.0, 0.25);
	const uint64_t batch_ends[] = {
		fine.sample_end,
		fine.sample_end + TEST_RING_SIZE,
	};
	struct panadapter_fft_frame *batch = wait_for_history_batch(context,
		&config, batch_ends, 2, 2);
	assert(batch[0].count == config.display_width_px);
	assert(batch[1].count == config.display_width_px);
	const double batch_first_peak = batch[0].first_hz +
		peak_bin(&batch[0]) * batch[0].bin_step_hz;
	const double batch_second_peak = batch[1].first_hz +
		peak_bin(&batch[1]) * batch[1].bin_step_hz;
	assert(fabs(batch_first_peak - 700.0) <= 1.5 * fabs(batch[0].bin_step_hz));
	assert(fabs(batch_second_peak + 700.0) <= 1.5 * fabs(batch[1].bin_step_hz));
	free(batch);
	assert_tone(context, 1000.0, 2500, 1250, &config, &generation);
	assert_tone(context, -1000.0, 2500, -1250, &config, &generation);
	assert_tone(context, 4500.0, 10000, 0, &config, &generation);
	assert_tone(context, 10000.0, 24980, 0, &config, &generation);

	config.display_span_hz = 2500;
	config.center_hz = 0;
	config.is_cw = 1;
	const int speeds[] = {12, 20, 50};
	for (unsigned index = 0; index < sizeof(speeds) / sizeof(speeds[0]); index++) {
		config.wpm = speeds[index];
		push_signal(context, 12000, 500.0, 0.25);
		const struct panadapter_fft_frame frame =
			wait_for_frame(context, &config, generation);
		generation = frame.generation;
		int expected = (int)lround((TEST_RATE / frame.decimation) * 1.2 / config.wpm);
		if (expected > frame.fft_bins) expected = frame.fft_bins;
		assert(frame.observation_samples == expected);
	}

	config.wpm = 20;
	push_signal(context, 7000, 500.0, 0.25);
	const struct panadapter_fft_frame keyed =
		wait_for_frame(context, &config, generation);
	generation = keyed.generation;
	push_signal(context, 7000, 0.0, 0.0);
	const struct panadapter_fft_frame gap =
		wait_for_frame(context, &config, generation);
	assert(peak_level(&keyed) - peak_level(&gap) > 40);

	config.is_tx = 1;
	struct panadapter_fft_frame stale_rx;
	assert(!panadapter_fft_get_frame(context, &config, &stale_rx));
	push_signal(context, 7000, 500.0, 0.25);
	const struct panadapter_fft_frame tx =
		wait_for_frame(context, &config, gap.generation);
	assert(tx.config.is_tx);

	struct panadapter_fft *const second_context = panadapter_fft_create();
	assert(second_context);
	struct panadapter_fft_config second_config = {
		.wpm = 20,
		.refresh_ms = 0,
		.display_width_px = 400,
	};
	uint64_t second_generation = 0;
	assert_tone(second_context, -700.0, 2500, 0, &second_config,
		&second_generation);
	panadapter_fft_destroy(second_context);
	struct panadapter_fft_frame primary_after_second;
	assert(panadapter_fft_get_frame(context, &config, &primary_after_second));
	assert(primary_after_second.generation == tx.generation);

	panadapter_fft_destroy(context);
	puts("panadapter FFT tests passed");
	return 0;
}
