#include "panadapter_view.h"

#include <math.h>

static void clamp_view(struct panadapter_view *view) {
  if (view->zoom < 1.0) view->zoom = 1.0;
  if (view->zoom > PANADAPTER_VIEW_MAX_ZOOM) view->zoom = PANADAPTER_VIEW_MAX_ZOOM;

  const double limit = 0.5 - 0.5 / view->zoom;
  if (view->center < -limit) view->center = -limit;
  if (view->center > limit) view->center = limit;
}

void panadapter_view_reset(struct panadapter_view *view) {
  view->zoom = 1.0;
  view->center = 0.0;
}

bool panadapter_view_is_default(const struct panadapter_view *view) {
  return fabs(view->zoom - 1.0) < 1.0e-6 && fabs(view->center) < 1.0e-6;
}

int panadapter_view_span_hz(const struct panadapter_view *view, int base_span_hz) {
  const int span = (int) lround(base_span_hz / view->zoom);
  return span > 0 ? span : 1;
}

int panadapter_view_center_hz(const struct panadapter_view *view, int base_span_hz) {
  return (int) lround(view->center * base_span_hz);
}

void panadapter_view_zoom_at(struct panadapter_view *view, double factor, double position) {
  const double old_zoom = view->zoom;
  view->zoom *= factor;
  clamp_view(view);
  view->center += (position - 0.5) * (1.0 / old_zoom - 1.0 / view->zoom);
  clamp_view(view);
}

void panadapter_view_pan(struct panadapter_view *view, double visible_fraction) {
  view->center += visible_fraction / view->zoom;
  clamp_view(view);
}

void panadapter_view_fit(struct panadapter_view *view, double start, double stop) {
  if (stop <= start)
    return;
  view->zoom = 1.0 / (stop - start);
  view->center = (start + stop) / 2.0 - 0.5;
  clamp_view(view);
}

double panadapter_view_map_position(const struct panadapter_view *old_view,
                                    const struct panadapter_view *new_view,
                                    double new_position) {
  const double frequency = new_view->center + (new_position - 0.5) / new_view->zoom;
  return 0.5 + (frequency - old_view->center) * old_view->zoom;
}
