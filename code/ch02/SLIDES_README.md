# Chapter 2 Slides

Interactive HTML slides for Chapter 2: GPU Hardware Architecture.

## Viewing the Slides

Open `ch02_slides.html` in any modern web browser:

```bash
# Option 1: Direct file open
open ch02_slides.html

# Option 2: Using Python HTTP server
python -m http.server 8000
# Then visit http://localhost:8000/ch02_slides.html
```

## Navigation

- **Keyboard**: Arrow keys ← → or Space to navigate
- **Touch**: Swipe left/right on mobile devices
- **Buttons**: Click Previous/Next buttons at the bottom

## Slide Content

1. **Title Slide** - Chapter overview with key metrics
2. **Problem Statement** - Why hardware architecture matters
3. **Memory Transfer** - PCIe vs Pinned memory comparison
4. **cuBLAS Optimization** - FP32 vs TF32
5. **Grace-Blackwell Coherency** - Zero-copy buffers and NVLink-C2C
6. **Hardware Inspection Tools** - Commands and utilities
7. **Blackwell Features** - B200/B300 capabilities
8. **Performance Results** - Measured deltas table
9. **Learning Goals** - Chapter objectives
10. **Directory Layout** - Code organization
11. **Validation Checklist** - Verification steps
12. **Summary** - Key takeaways

## Key Metrics Highlighted

- **Grace Coherent Memory**: 23.14x speedup
- **Memory Transfer**: 5.20x speedup  
- **cuBLAS GEMM**: 5.17x speedup

## Technical Details

The slides are self-contained HTML with:
- CSS animations and transitions
- Responsive design for mobile/desktop
- Keyboard and touch navigation
- Progress bar indicator
- Code syntax highlighting
- Comparison tables and metric cards