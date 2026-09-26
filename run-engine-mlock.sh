#!/bin/bash
# Wrapper: the engine carries cap_ipc_lock + cap_sys_nice (file capabilities, so no sudo is needed at
# run time), which is what lets `pin_hot` actually mlock the host tier instead of leaving it reclaimable.
# nice -5 keeps the pool/reader threads from being starved by the desktop.
exec nice -n -5 /home/bodhi/models/strata/engine/strata "$@"
