#!/bin/bash
# Quick test progress checker

echo "📊 Test Progress Monitor"
echo "======================="

# Check if test is running
if pgrep -f "test_runner.py" > /dev/null; then
    echo "✅ Test runner is running"

    # Get latest test number from PM2 logs or ps
    LATEST=$(pm2 logs ebook-python-api --lines 20 --nostream 2>/dev/null | grep -oP '\[(\d+)/60\]' | tail -1)
    if [ -n "$LATEST" ]; then
        echo "Latest test: $LATEST"
    fi
else
    echo "❌ Test runner not running"
fi

# Check for result files
echo ""
echo "Result files:"
ls -lht test_results_*.json 2>/dev/null | head -3 || echo "No result files yet"

# Check background task
echo ""
echo "Background tasks:"
ps aux | grep -E "test_runner|python.*test" | grep -v grep | head -3

echo ""
echo "To view live output: tail -f test_output_final.log"
echo "Or check task output: cat /tmp/claude/-workspace-ebook-pdf-ai-chatbot-sp-2025-main/tasks/bfedba6.output | tail -100"
