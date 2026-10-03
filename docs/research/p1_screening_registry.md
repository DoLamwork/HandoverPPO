# Registry screening P1 E00-E03

Nguồn: artifact validation trong `results/models/ablation_v2`, tổng hợp ngày
03/10/2026 bằng command `summarize_ablation`. Không arm nào evaluate test hoặc
zero-shot.

| ID | K | Order | Status | Selected step | Last validation | RR selected | RLF-time | HOF | PP | STAY | C1 | C2+ | Selected mode | Last mode |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| E00_f1_k10_legacy_s17 | 10 | legacy | QOS_FEASIBLE | 92.160 | 92.160 | 91,272% | 1,313% | 10,903% | 37,057% | 0% | 100% | 0% | HO_C1 collapse | HO_C1 collapse |
| E01_ranked_k10_s17 | 10 | ranked_v1 | QOS_INFEASIBLE | 10.240 | 92.160 | 74,004% | 1,718% | 0% | 0% | 99,89% | 0,11% | 0% | STAY collapse | STAY collapse |
| E02_ranked_k5_s17 | 5 | ranked_v1 | QOS_FEASIBLE | 51.200 | 92.160 | 91,272% | 1,313% | 10,903% | 37,057% | 0% | 100% | 0% | HO_C1 collapse | STAY collapse |
| E03_ranked_k3_s17 | 3 | ranked_v1 | QOS_FEASIBLE | 92.160 | 92.160 | 91,272% | 1,313% | 10,903% | 37,057% | 0% | 100% | 0% | HO_C1 collapse | HO_C1 collapse |

## Quyết định

K*=3/ranked_v1 được chọn làm nền P2 vì:

1. K=3 giữ 100% alternative hữu ích trong positive state của audit train-only.
2. Selected metric của K=3 không kém K=5/K=10 trong screening seed17.
3. Observation giảm từ 109 xuống 39 và action giảm từ 11 xuống 4.
4. Khi hiệu năng chưa phân biệt, kế hoạch preregister ưu tiên K nhỏ hơn.

Quyết định này chỉ khóa shortlist cho ablation observation. Nó không chứng minh
K=3 tốt hơn về vận hành: checkpoint feasible của K=3 vẫn chỉ sao chép 3GPP bằng
100% HO_C1, và chưa policy nào học dùng C2/C3.
