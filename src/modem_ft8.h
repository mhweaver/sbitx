#define FT8_MAX_BUFF (12000 * 18)
#define FTX_QUEUE_CAP 16 // hard ceiling on the FTx caller queue; also main_controls[]'s FTX_QUEUE_MAX max

void ft8_rx(int32_t *samples, int count);
void ft8_init();
void ft8_abort(bool terminate_qso);
void ft8_tx(char *message, int freq);
void ft8_tx_3f(const char* call_to, const char* call_de, const char* extra);
void ft8_poll(int tx_is_on);
float ft8_next_sample();
void ft8_call(int sel_time);
void ftx_call_or_continue(const char* line, int line_len, const text_span_semantic* spans, bool is_click);
// Exposed (non-static) purely so tests can call this pure timing check deterministically,
// without needing to fake the wallclock.
bool ftx_slot_has_room(int slot_relative_ms, bool is_ft4);

// Exposed (non-static) purely so tests can introspect the FTx caller queue's contents/order
// without a general-purpose API: rank 0 is the next callsign ftx_queue_dequeue_next() would act on.
int ftx_queue_count(void);
const char *ftx_queue_callsign_at(int rank);
// Exposed (non-static) purely so tests can clear the queue between independent test cases.
void ftx_queue_reset_for_test(void);
