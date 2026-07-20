#ifndef PANADAPTER_VIEW_H
#define PANADAPTER_VIEW_H

#include <stdbool.h>

#define PANADAPTER_VIEW_MAX_ZOOM 16.0

struct panadapter_view {
  double zoom;
  double center;
};

void panadapter_view_reset(struct panadapter_view *view);
bool panadapter_view_is_default(const struct panadapter_view *view);
int panadapter_view_span_hz(const struct panadapter_view *view, int base_span_hz);
int panadapter_view_center_hz(const struct panadapter_view *view, int base_span_hz);
void panadapter_view_zoom_at(struct panadapter_view *view, double factor, double position);
void panadapter_view_pan(struct panadapter_view *view, double visible_fraction);

#endif
