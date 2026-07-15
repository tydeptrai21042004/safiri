# Data card

## Dataset status

The default dataset is fully synthetic. It is generated for the SAFiRi three-day take-home assignment and must not be represented as SAFiRi operational data.

## Scope

- 300 shipments by default.
- Six route profiles.
- Sea, air, road and multimodal modes.
- Eight canonical milestones from booking to final delivery.
- UTC timestamps.
- Complete ground truth plus a separate observed-event stream.

## Generation mechanism

Planned times come from route-specific stage schedules. Actual stage duration depends on congestion, weather, document readiness, cargo type, service level, random disruptions and partial recovery from existing delay. Because each actual stage begins after the previous one ends, upstream delay propagates downstream naturally.

Observed events include reporting lag, source reliability, missing intermediate milestones and duplicates. `actual_time` records the physical event time; `observed_at` records when the system received it.

## Intended use

- Demonstrate point-in-time ETA prediction.
- Compare baseline, direct and stage-aware models.
- Test missing and late event handling.
- Build an analyst-facing explanation demo.

## Not intended for

- Production accuracy claims.
- Carrier comparison.
- Pricing, contracting or automated shipment rerouting.
- Claims about real customs, ports or SAFiRi customers.

## Reproducibility

The random seed and simulation parameters are in `configs/default.yaml`. Run `make all` to recreate every data and model artifact.

