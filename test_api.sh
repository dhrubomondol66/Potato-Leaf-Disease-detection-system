#!/bin/bash
# Simple test of the API
echo "Testing health endpoint..."
curl -v "http://127.0.0.1:8000/health" 2>&1 | head -20

echo ""
echo "Testing home endpoint..."
curl -s "http://127.0.0.1:8000/" 2>&1 | head -20
