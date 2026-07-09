#ifndef DEEPFILTER_ANR_H
#define DEEPFILTER_ANR_H

#include <stdint.h>

void deepfilter_anr_reset(void);
int deepfilter_anr_process(int32_t *samples, int n_samples);

#endif
