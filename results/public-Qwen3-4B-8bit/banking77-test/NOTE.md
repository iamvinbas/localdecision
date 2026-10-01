# Latency not comparable

This run was paused at regular intervals (SIGSTOP / SIGCONT) to keep the laptop's GPU at about
half load, so the `request_ms` values in these records include the pauses and are left out of
the README. Accuracy, log-probabilities and calibration are unaffected: the model computes the
same numbers whether or not it is paused.
