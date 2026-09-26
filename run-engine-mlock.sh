#!/bin/bash
# Wrapper: raise RLIMIT_MEMLOCK (for the expert hot tier's mlock) via scoped sudo,
# then drop back to the invoking user and exec the real engine.  The engine runs at
# nice -5 (set as root before the UID drop) so the pool/reader threads are not starved
# by the desktop under memory or CPU pressure.
if [ "${STRATA_MEMLOCK_DONE}" != "1" ]; then
  exec sudo -n /usr/bin/prlimit --memlock=unlimited:unlimited -- \
    nice -n -5 \
    setpriv --reuid="$(id -u)" --regid="$(id -g)" --init-groups -- \
    env STRATA_MEMLOCK_DONE=1 LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" /home/bodhi/models/strata/engine/strata "$@"
fi
exec /home/bodhi/models/strata/engine/strata "$@"
