#!/bin/bash
# Simple script to open Chapter 2 slides in browser

echo "Opening Chapter 2: GPU Hardware Architecture slides..."
echo ""

# Check if file exists
if [ ! -f "ch02_slides.html" ]; then
    echo "Error: ch02_slides.html not found"
    exit 1
fi

# Try to open in default browser
if command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open ch02_slides.html
elif command -v open &> /dev/null; then
    # macOS
    open ch02_slides.html
elif command -v start &> /dev/null; then
    # Windows
    start ch02_slides.html
else
    echo "Could not detect browser opener. Please open ch02_slides.html manually."
    echo ""
    echo "Alternative: Start a local HTTP server:"
    echo "  python3 -m http.server 8000"
    echo "  Then visit: http://localhost:8000/ch02_slides.html"
fi

echo ""
echo "Navigation:"
echo "  - Arrow keys ← → or Space to navigate"
echo "  - Touch: Swipe left/right on mobile"
echo "  - Buttons: Click Previous/Next at bottom"
echo ""
echo "Enjoy the slides!"