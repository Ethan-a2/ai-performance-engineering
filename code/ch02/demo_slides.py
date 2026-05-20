#!/usr/bin/env python3
"""
Demo script to showcase Chapter 2 slides functionality.
This script provides a quick way to verify the slides work correctly.
"""

import webbrowser
import os
import sys
from pathlib import Path


def main():
    """Main demo function."""
    print("=== Chapter 2: GPU Hardware Architecture Slides Demo ===\n")

    # Check if slides file exists
    slides_file = Path("ch02_slides.html")
    if not slides_file.exists():
        print("❌ Error: ch02_slides.html not found in current directory")
        print("Please run this script from the ch02 directory.")
        sys.exit(1)

    print("✅ Found ch02_slides.html")
    print(f"   File size: {slides_file.stat().st_size / 1024:.1f} KB")

    # Count slides
    with open(slides_file, "r") as f:
        content = f.read()
        slide_count = content.count('id="slide')
        print(f"   Total slides: {slide_count}")

    print("\n📊 Slide Overview:")
    print("   1. Title - Key metrics (23.14x, 5.20x, 5.17x speedups)")
    print("   2. Problem - Why hardware architecture matters")
    print("   3. Memory Transfer - PCIe vs Pinned memory")
    print("   4. cuBLAS - FP32 vs TF32 optimization")
    print("   5. Grace-Blackwell - Coherent memory features")
    print("   6. Tools - Hardware inspection commands")
    print("   7. Blackwell - B200/B300 capabilities")
    print("   8. Results - Performance comparison table")
    print("   9. Goals - Chapter learning objectives")
    print("  10. Layout - Directory structure")
    print("  11. Validation - Verification checklist")
    print("  12. Summary - Key takeaways")

    print("\n🎯 Key Features:")
    print("   • Responsive design (desktop + mobile)")
    print("   • Keyboard navigation (arrow keys, space)")
    print("   • Touch/swipe support")
    print("   • Progress bar indicator")
    print("   • Code syntax highlighting")
    print("   • Interactive comparison tables")
    print("   • Performance metric cards")

    print("\n🚀 Opening slides in browser...")

    try:
        # Get absolute file path
        abs_path = slides_file.resolve()
        file_url = f"file://{abs_path}"

        # Try to open in browser
        webbrowser.open(file_url)
        print(f"✅ Opened: {file_url}")

        print("\n📖 Navigation Instructions:")
        print("   • Use arrow keys ← → to navigate")
        print("   • Press Space for next slide")
        print("   • Click Previous/Next buttons")
        print("   • Swipe on mobile devices")
        print("   • Progress bar shows current position")

        print("\n💡 Tips:")
        print("   • Press 'F' for fullscreen (browser dependent)")
        print("   • Use Ctrl+/- to zoom in/out")
        print("   • Print with Ctrl+P for PDF export")

    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print("\nManual options:")
        print(f"   1. Open {abs_path} directly in browser")
        print("   2. Run: python3 -m http.server 8000")
        print("      Then visit: http://localhost:8000/ch02_slides.html")

    print("\n" + "=" * 60)
    print("Enjoy exploring Chapter 2: GPU Hardware Architecture!")
    print("=" * 60)


if __name__ == "__main__":
    main()
