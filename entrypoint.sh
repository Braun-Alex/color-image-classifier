#!/bin/bash

mpirun --allow-run-as-root --oversubscribe -n $(nproc) ./color_image_classifier