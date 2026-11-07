# NPR 7150.2D data

This directory contains NASA's **NPR 7150.2D, NASA Software Engineering
Requirements**, plus derived files suitable for retrieval and evaluation code.

## Files

- `npr-7150-2d.pdf` — complete 89-page PDF downloaded from an official NASA
  Langley mirror.
- `npr-7150-2d-swehb.html` — full HTML copy downloaded from NASA's Software
  Engineering Handbook site.
- `npr-7150-2d.txt` — plain-text extraction of the HTML copy; a source/reference
  artifact, not the canonical pipeline's ingestion input.
- **`npr-7150-2d-requirements_fixed.csv`** — canonical input selected by
  `src/config.py`: 130 SWE records with complete repaired clause-local text,
  including attached lists, conditions, notes, and a traceability table.
- `npr-7150-2d-requirements_fixed.json` — equivalent repaired records as JSON;
  checked against the CSV by the corpus integration tests.
- `npr-7150-2d-requirements.csv` and `.json` — preserved original extraction,
  used by the October workflow. These omit text restored in 62 records and should
  not replace the repaired input in the canonical workflow.

The repair expanded text in **62 records**, left **68 unchanged**, and added no
new records or SWE IDs. Restoring a note does not make it an additional mandatory
obligation. See the [corpus-repair guide](../docs/nasa-corpus-repair.md) for a
before/after example, source-page inventory, verification, and index recovery.

Official sources:

- NASA SWEHB full text: <https://swehb.nasa.gov/spaces/SITE/pages/123601159/NPR%2B7150.2D>
- NASA Langley PDF mirror: <https://explorers.larc.nasa.gov/APSMEX26/SMEX/pdf_files/NASA21_NPR_7150_2D.pdf>
- NODIS record: <https://nodis3.gsfc.nasa.gov/displayDir.cfm?c=7150&s=2D&t=NPR>

Benchmark:

- `npr7150_ground_truth_benchmark.csv` — human-labeled requirement/clause pairs
  used by the October baseline. Renamed from the duplicated upload filename;
  its contents are unchanged.

The benchmark contains 50 project requirements with intended NASA clauses,
`Meets`/`Partial`/`Gap` labels, and rationales. Labels have not been changed by the
corpus repair. The set has been reused during development and is not an independent
holdout; applicability and clause-choice ambiguities remain. See the
[current results and limitations](../README.md#verified-benchmark-and-limitations).
