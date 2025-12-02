# Project Overview

Cod v2 consumes the report structure defined by Cod Analyzer in [Precise Compositional Buffer Overflow Detection via Heap Disjointness](https://doi.org/10.1145/3650212.3652110) and emits an augmented report with added context and targeted hints.

# Directory Structure

`main.py` — entry point; processes a source directory and a JSON report.  
`BufferTrace.py` — aligns source locations with report entries.  
`final_report_[tcpdump|zstd].json` — original Cod Analyzer reports.  
`final_report_[tcpdump|zstd]_v2.json` — Cod v2 enhanced reports.  
`final_report_zstd_bloat.json` — example of excessive detail leading to report bloat.  

# Requirements

Python 3  
`pycparser`  

# Usage

```
python3 main.py <src-dir> <report.json>
```

`<src-dir>` contains the sources corresponding to the bitcode used for the original report.
`<report.json>` is the Cod Analyzer output.

# Output

Enhanced report with added context and refined hints, written to `final_report_<src-dir>_v2.json`.

