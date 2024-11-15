#!/bin/bash

echo "Opening connection..."
ssh -f -N -L 5000:localhost:5000 maxspad@greatlakes.arc-ts.umich.edu

