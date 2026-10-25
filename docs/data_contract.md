# MaverickX Data Contract (foundation version)

## Internal batch table

Every row is one historical measurement time. Required fields are:

| Field | Meaning | Unit |
|---|---|---|
| `batch_id` | Stable identifier for one completed batch | text |
| `elapsed_time_h` | Hours since batch start | h |
| `dissolved_oxygen_mg_l` | Dissolved oxygen concentration | mg/L |
| `ph` | Acidity/alkalinity measurement | pH |
| `temperature_k` | Broth temperature | K |
| `pressure_bar` | Vessel head pressure | bar |

Optional online process fields include aeration, agitation, substrate feed, acid/base flow, coolant/heating flow, oil/PAA flow, off-gas measurements, OUR, CER, volume, weight, substrate, and generated heat. The loader preserves a unit map in batch metadata.

The `isolation_forest_v1` artifact intentionally uses only the four required measurements, so a valid minimum-input batch remains analyzable. Optional channels are available for later extended-model comparisons.

Ground-truth fields such as `fault_label`, `fault_active`, and `fault_intervals` are evaluation metadata. They are never included in the feature matrix.

## Analysis timeline output

Each eligible rolling window contains:

- `batch_id`
- `window_start_h`
- `window_end_h`
- `anomaly_score` (larger means more unusual)
- `alert_threshold`
- `is_flagged`
- `model_version`

## JSON summary

The JSON output contains batch metadata, data-quality information, model identity, all timeline records, grouped anomaly events, and explicit limitations. It may contain a separate `evaluation_ground_truth` section when known simulator labels are available; that section is not needed for ordinary inference.

