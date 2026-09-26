#!/bin/sh
cd "/home/bodhi/models/strata"
exec "/home/bodhi/models/strata/venv/bin/python" "/home/bodhi/models/strata/serve/server.py" --engine strata --config "/home/bodhi/models/strata/strata-iq4_xs.json" --port 8103
