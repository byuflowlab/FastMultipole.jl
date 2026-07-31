# Task 023 integration benchmark — mecsrs-MacBook-Pro-188.local, 2026-07-14T23:15:07.216

Julia 1.12.5, threads=1, gpu=false

| backend | path | n | ell | P | phase | seconds |
|---|---|---|---|---|---|---|
| cpu | radix | 10000 | 4 | 4 | construct | 0.59627 |
| cpu | radix | 10000 | 4 | 4 | step | 5.34566 |
| cpu | legacy | 10000 | 4 | 4 | step | 0.19167 |
| cpu | direct | 10000 | 4 | 4 | step | 0.28958 |
| cpu | radix | 100000 | 4 | 4 | construct | 0.22264 |
| cpu | radix | 100000 | 4 | 4 | step | 20.17752 |
| cpu | legacy | 100000 | 4 | 4 | step | 3.54961 |
