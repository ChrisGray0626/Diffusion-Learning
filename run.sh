#!/bin/bash
docker rmi downscale-sm:latest 2>/dev/null || true && docker-compose build && docker-compose up -d
