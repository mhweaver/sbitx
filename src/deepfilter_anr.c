#include "deepfilter_anr.h"
#include "sound.h"

#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DF_AUDIO_SCALE 2147483648.0f
#define DF_MAX_FRAME 4096
#define DF_READY_CAP (DF_MAX_FRAME * 4)

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

static int path_is_absolute(const char *path)
{
	return path && path[0] == '/';
}

static int get_exe_dir(char *dir, size_t dir_size)
{
#ifdef __linux__
	char exe_path[PATH_MAX];
	ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
	if (len <= 0)
		return 0;
	exe_path[len] = '\0';
	char *slash = strrchr(exe_path, '/');
	if (!slash)
		return 0;
	*slash = '\0';
	snprintf(dir, dir_size, "%s", exe_path);
	return 1;
#else
	(void)dir;
	(void)dir_size;
	return 0;
#endif
}

static int make_exe_relative_path(const char *path, char *resolved, size_t resolved_size)
{
	char exe_dir[PATH_MAX];
	if (path_is_absolute(path) || !get_exe_dir(exe_dir, sizeof(exe_dir)))
		return 0;
	snprintf(resolved, resolved_size, "%s/%s", exe_dir, path);
	return 1;
}

static int try_dlopen(const char *path)
{
	char resolved[PATH_MAX];
	if (!path || !path[0])
		return 0;
	df_lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (df_lib)
		return 1;
	if (make_exe_relative_path(path, resolved, sizeof(resolved))) {
		df_lib = dlopen(resolved, RTLD_NOW | RTLD_LOCAL);
		if (df_lib)
			return 1;
	}
	return 0;
}

static const char *resolve_data_path(const char *path, char *resolved, size_t resolved_size)
{
	if (path_is_absolute(path))
		return path;
	if (access(path, R_OK) == 0)
		return path;
	if (make_exe_relative_path(path, resolved, resolved_size) && access(resolved, R_OK) == 0)
		return resolved;
	return path;
}

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
	const char *env_lib = getenv("SBITX_DEEPFILTER_LIB");
	const char *candidates[] = {
		"ext/DeepFilterNet/target/release/libdf.so",
		"ext/DeepFilterNet/target/release/libdeep_filter.so",
		"ext/DeepFilterNet/target/release/libdeepfilter.so",
		NULL,
	};

	if (df_state)
		return 1;

	if (!df_lib) {
		try_dlopen(env_lib);
		for (int i = 0; candidates[i]; i++) {
			if (try_dlopen(candidates[i]))
				break;
		}
		if (!df_lib) {
			if (!df_reported_missing) {
				fprintf(stderr, "DeepFilterNet: shared library not found; run ./build sbitx first\n");
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

	char resolved_model_path[PATH_MAX];
	const char *model_path = resolve_data_path(deepfilter_model_path, resolved_model_path,
	                                           sizeof(resolved_model_path));
	df_state = p_df_create(model_path, (float)deepfilter_atten_lim);
	if (!df_state) {
		fprintf(stderr, "DeepFilterNet: could not create model from %s\n", model_path);
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

static float clamp_float(float v, float lo, float hi)
{
	if (v < lo)
		return lo;
	if (v > hi)
		return hi;
	return v;
}

static int32_t float_to_i32(float v)
{
	v = clamp_float(v, -1.0f, 1.0f);
	return (int32_t)(v * DF_AUDIO_SCALE);
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
			int32_t s = float_to_i32(output);
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
