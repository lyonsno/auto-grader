These PNG fixtures are rasterized at 200 DPI from the real scanner-produced
PDF `auto-grader-assets/scans/end_to_end_demo_2026_04_12/04122026.pdf`
(a 5-page generated MC exam scanned on an institutional scanner on 2026-04-12).

At 200 DPI, pages 3 and 5 fall in OpenCV's QR detector blind spot at native
resolution and require multi-scale retry to decode. This is the exact failure
mode that motivated the QR readback hardening work.

Do not regenerate these fixtures from the PDF without also re-verifying which
pages require rescale — the pathology is DPI-dependent.
