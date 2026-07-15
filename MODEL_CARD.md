# Model card

## Models

1. **Route-stage median baseline**: median duration for each route and target milestone.
2. **Direct ETA model**: gradient boosting predicts total remaining hours.
3. **Stage-aware ETA model**: one gradient-boosting model predicts each remaining stage duration; predictions are summed.
4. **Delay-risk model**: class-weighted gradient-boosting classifier with validation-set probability calibration.

## Inputs

Only point-in-time features are used: route, mode, current milestone, current schedule delay, event age, congestion, weather, document readiness, missing/late/duplicate counts, source reliability and booking-time features.

## Outputs

- ETA P10, P50 and P90.
- Remaining hours and predicted delay versus plan.
- Probability of exceeding the 12-hour delay threshold.
- Remaining-stage breakdown.
- Local perturbation drivers.
- Data-quality score, exceptions and recommended actions.

## Evaluation

Shipments are ordered by booking time and divided 70/15/15 into train, validation and test. Every snapshot from one shipment remains in the same split. The test set is not used for training, calibration or interval fitting.

## Explainability warning

Local perturbation measures how a prediction changes when one feature is replaced by its training reference. It describes model sensitivity and is not a causal estimate.

## Limitations

- Synthetic training and testing data.
- Small assignment-sized sample.
- Unseen route performance is not guaranteed.
- Uncertainty intervals can become miscalibrated under distribution shift.
- Recommendations are decision support and require human review.

