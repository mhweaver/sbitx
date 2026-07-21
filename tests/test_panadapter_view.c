#include "panadapter_view.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
  struct panadapter_view view;
  panadapter_view_reset(&view);
  assert(panadapter_view_is_default(&view));
  assert(panadapter_view_span_hz(&view, 25000) == 25000);

  panadapter_view_zoom_at(&view, 2.0, 0.75);
  assert(panadapter_view_span_hz(&view, 25000) == 12500);
  assert(panadapter_view_center_hz(&view, 25000) == 3125);
  const struct panadapter_view full = {1.0, 0.0};
  assert(fabs(panadapter_view_map_position(&full, &view, 0.0) - 0.375) < 1.0e-9);
  assert(fabs(panadapter_view_map_position(&full, &view, 1.0) - 0.875) < 1.0e-9);

  panadapter_view_reset(&view);
  panadapter_view_zoom_at(&view, 2.0, 0.75);
  panadapter_view_pan(&view, 0.25);
  assert(panadapter_view_span_hz(&view, 25000) == 12500);
  assert(panadapter_view_center_hz(&view, 25000) == 6250);

  panadapter_view_pan(&view, 10.0);
  assert(fabs(view.center - 0.25) < 1.0e-9);
  panadapter_view_zoom_at(&view, 0.01, 0.5);
  assert(panadapter_view_is_default(&view));

  panadapter_view_zoom_at(&view, 1000.0, 0.5);
  assert(view.zoom == PANADAPTER_VIEW_MAX_ZOOM);
  assert(panadapter_view_span_hz(&view, 25000) == 50);

  panadapter_view_fit(&view, 0.504, 0.592);
  assert(panadapter_view_span_hz(&view, 25000) == 2200);
  assert(panadapter_view_center_hz(&view, 25000) == 1200);

  assert(panadapter_grid_step_hz(25000) == 2500);
  assert(panadapter_grid_step_hz(12500) == 1000);
  assert(panadapter_grid_step_hz(2200) == 200);
  assert(panadapter_grid_step_hz(50) == 5);
  assert(panadapter_grid_first_hz(7087500, 2500) == 7087500);
  assert(panadapter_grid_first_hz(7087600, 2500) == 7090000);
  assert(panadapter_grid_first_hz(-1200, 500) == -1000);
  assert(panadapter_grid_label_hz(14074750, 3000, 250) == 14074800);
  assert(panadapter_grid_label_hz(14074750, 25000, 2500) == 14075000);

  puts("panadapter view tests passed");
  return 0;
}
