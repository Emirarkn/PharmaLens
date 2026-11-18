# Units and value contract — v001 (DRAFT)

Status: **draft_unapproved**. Monetary metrics on real data stay off without finance/commercial and data-owner approval (gate G1).

| Field | Definition (draft) | Source | Note |
|---|---|---|---|
| `units_paid` | Invoiced (paid) boxes | reconciliation `SatisAdet` | Integer, cannot be negative; negative → quarantine |
| `units_free` | Free-goods (bonus) boxes | reconciliation `MFAdet` | Empty = unknown (NULL), not zero |
| `returns_units` | Returned boxes | reconciliation `IadeAdet` | Kept separate; never deleted |
| `net_units` | `units_paid + units_free − returns_units` | canonical | A net-negative month is not an error → `observed_net_negative` |
| `units_total` (VENDOR) | VENDOR's single box measurement | VENDOR `Units` | No paid/free split; **compared with** reconciliation `net_units`, never summed |
| `net_value_vat_inc_try` | VAT-inclusive net value (TRY, decimal) | reconciliation `NetTutarKDVDahil` | VAT is **not added again**; the decimal separator is set in the manifest/YAML (`,`) |

Rules:
- No box → treatment conversion is made until barcode strength/dose equivalence is verified.
- If the value field cannot be verified, the related analysis continues on units only (a written decision is required).
- The synthetic demo uses the draft above; on real data the first job is a sample reconciliation (a few raw rows are computed by hand with the source owner).
