# EXP086 producer review note

Observed source exists; evaluator development is continuing against exp086-dinov2s-v1.

- C01: pilot step timer starts after DataLoader yield. End-to-end steady intervals should include data fetch/wait, otherwise full wall-time projection can be optimistic.
- C02: finish_pilot uses 1.3*(steady*3448)+300; cost gate exact comparison uses (1.3*steady)*3448+300. Use one expression or explicit roundoff tolerance.
- C03: runner loads V label values, then substitutes zero V targets. The valid claim is no V metrics/intermediate V inference, not never-read V labels.

Do not freeze registration until C01/C02 are resolved in producer source/protocol. Current7 statistical synthetic checks pass; no real evaluation or GPU work.
