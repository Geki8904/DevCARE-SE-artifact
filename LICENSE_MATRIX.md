# License matrix

DevCARE-SE separates software licensing from research-artifact licensing so
that reuse terms remain unambiguous.

| Material | Included paths | License | Required action |
|---|---|---|---|
| Analysis and reproduction code | `code/**` | MIT | Retain the MIT copyright and permission notice. |
| Environment specification | `requirements.txt` | MIT | Retain the MIT copyright and permission notice. |
| De-identified modeling data | `data/**` | CC BY 4.0 | Credit the DevCARE-SE authors, link the license, and identify modifications. |
| Aggregate experiment and audit outputs | `results/**` | CC BY 4.0 | Credit the DevCARE-SE authors, link the license, and identify modifications. |
| Exported research figures | `figures/**` | CC BY 4.0 | Credit the DevCARE-SE authors, link the license, and identify modifications. |
| Package documentation and citation metadata | `README.md`, `DATA_DICTIONARY.md`, `CITATION.cff` | CC BY 4.0 | Credit the DevCARE-SE authors, link the license, and identify modifications. |
| Integrity manifest | `MANIFEST.sha256` | CC BY 4.0 | Preserve provenance when redistributing the package. |

## Exclusions

The package does not distribute or license private survey responses,
individual expert allocations, annotator workbooks, credentials, raw database
exports, repository identity mappings, exact PR identifiers, or exact event
timestamps. Third-party libraries remain under their respective licenses.
The manuscript and publisher-formatted paper are not included in this license
matrix.
