#include "deepfilter_anr.h"
#include "sound.h"

#include <dlfcn.h>
#include <math.h>
#include <stdio.h>

#define DF_AUDIO_SCALE 2147483648.0f
#define DF_MAX_FRAME 4096
#define DF_READY_CAP (DF_MAX_FRAME * 4)
#define DF_LIB_PATH "ext/DeepFilterNet/target/release/libdf.so"
#define DF_MODEL_PATH "ext/DeepFilterNet/models/DeepFilterNet3_ll_onnx.tar.gz"

struct df_state;

typedef struct df_state *(*df_create_fn)(const char *, float);
typedef size_t (*df_get_frame_length_fn)(struct df_state *);
typedef float (*df_process_frame_fn)(struct df_state *, float *, float *);
typedef void (*df_set_atten_lim_fn)(struct df_state *, float);
typedef void (*df_set_post_filter_beta_fn)(struct df_state *, float);
typedef void (*df_free_fn)(struct df_state *);

static void *df_lib;
static struct df_state *df_state;
static df_create_fn p_df_create;
static df_get_frame_length_fn p_df_get_frame_length;
static df_process_frame_fn p_df_process_frame;
static df_set_atten_lim_fn p_df_set_atten_lim;
static df_set_post_filter_beta_fn p_df_set_post_filter_beta;
static df_free_fn p_df_free;

static size_t df_frame_len;
static float df_in[DF_MAX_FRAME];
static float df_out[DF_MAX_FRAME];
static float df_ready[DF_READY_CAP];
static int df_fill;
static int df_ready_read;
static int df_ready_count;
static int df_reported_missing;

static int last_atten_lim = -1;
static int last_pf_beta = -1;

static int load_symbol(void **target, const char *name)
{
	*target = dlsym(df_lib, name);
	if (!*target) {
		fprintf(stderr, "DeepFilterNet: missing symbol %s: %s\n", name, dlerror());
		return 0;
	}
	return 1;
}

static int deepfilter_anr_init(void)
{
	if (df_state)
		return 1;

	if (!df_lib) {
		df_lib = dlopen(DF_LIB_PATH, RTLD_NOW | RTLD_LOCAL);
		if (!df_lib) {
			if (!df_reported_missing) {
				fprintf(stderr, "DeepFilterNet: %s not found; run ./build sbitx first\n",
				        DF_LIB_PATH);
				df_reported_missing = 1;
			}
			return 0;
		}

		if (!load_symbol((void **)&p_df_create, "df_create") ||
		    !load_symbol((void **)&p_df_get_frame_length, "df_get_frame_length") ||
		    !load_symbol((void **)&p_df_process_frame, "df_process_frame") ||
		    !load_symbol((void **)&p_df_set_atten_lim, "df_set_atten_lim") ||
		    !load_symbol((void **)&p_df_set_post_filter_beta, "df_set_post_filter_beta") ||
		    !load_symbol((void **)&p_df_free, "df_free")) {
			dlclose(df_lib);
			df_lib = NULL;
			return 0;
		}
	}

	df_state = p_df_create(DF_MODEL_PATH, (float)deepfilter_atten_lim);
	if (!df_state) {
		fprintf(stderr, "DeepFilterNet: could not create model from %s\n", DF_MODEL_PATH);
		return 0;
	}

	df_frame_len = p_df_get_frame_length(df_state);
	if (df_frame_len == 0 || df_frame_len > DF_MAX_FRAME) {
		fprintf(stderr, "DeepFilterNet: unsupported frame length %zu\n", df_frame_len);
		p_df_free(df_state);
		df_state = NULL;
		df_frame_len = 0;
		return 0;
	}

	last_atten_lim = -1;
	last_pf_beta = -1;
	df_fill = 0;
	return 1;
}

void deepfilter_anr_reset(void)
{
	if (df_state) {
		p_df_free(df_state);
		df_state = NULL;
	}
	df_fill = 0;
	df_ready_read = 0;
	df_ready_count = 0;
	last_atten_lim = -1;
	last_pf_beta = -1;
}

static void deepfilter_update_controls(void)
{
	if (deepfilter_atten_lim != last_atten_lim) {
		p_df_set_atten_lim(df_state, (float)deepfilter_atten_lim);
		last_atten_lim = deepfilter_atten_lim;
	}
	if (deepfilter_pf_beta != last_pf_beta) {
		p_df_set_post_filter_beta(df_state, (float)deepfilter_pf_beta / 1000.0f);
		last_pf_beta = deepfilter_pf_beta;
	}
}

static void deepfilter_push_output(float sample)
{
	int write = (df_ready_read + df_ready_count) % DF_READY_CAP;
	df_ready[write] = sample;
	if (df_ready_count < DF_READY_CAP) {
		df_ready_count++;
	} else {
		df_ready_read = (df_ready_read + 1) % DF_READY_CAP;
	}
}

static int deepfilter_pop_output(float *sample)
{
	if (df_ready_count == 0)
		return 0;
	*sample = df_ready[df_ready_read];
	df_ready_read = (df_ready_read + 1) % DF_READY_CAP;
	df_ready_count--;
	return 1;
}

int deepfilter_anr_process(int32_t *samples, int n_samples)
{
	if (!deepfilter_anr_init())
		return 0;

	deepfilter_update_controls();

	for (int i = 0; i + 1 < n_samples; i += 2) {
		float input = ((float)samples[i] + (float)samples[i + 1]) / (2.0f * DF_AUDIO_SCALE);
		float output;

		if (deepfilter_pop_output(&output)) {
			double clipped = fmin(fmax((double)output, -1.0), 1.0);
			int32_t s = (int32_t)lrint(clipped * 2147483647.0);
			samples[i] = s;
			samples[i + 1] = s;
		}

		df_in[df_fill++] = input;
		if ((size_t)df_fill == df_frame_len) {
			p_df_process_frame(df_state, df_in, df_out);
			df_fill = 0;
			for (size_t j = 0; j < df_frame_len; j++)
				deepfilter_push_output(df_out[j]);
		}
	}

	return 1;
}
