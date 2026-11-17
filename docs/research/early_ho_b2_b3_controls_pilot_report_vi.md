# Early-HO T1 — pilot B2 và B3

Ngày: 10/10/2026. Nhánh: `codex/early-ho-timing`.
Artifact: `results/analysis/early_ho_timing/b2_b3_controls_pilot_v1`.
Trạng thái: **PILOT COMPLETE; chưa đánh giá gate bằng CI**.

## Contract đã triển khai

- B2: khi best-neighbor RSRP L3 − serving RSRP L3 `>1 dB`, HO ngay tới
  best neighbor tại tick decision.
- B3: cùng early window, lên lịch HO sau 40 ms (4 tick). Nếu trước lúc thực
  thi delta `<=0 dB`, serving đổi hoặc best target đổi thì hủy lịch cũ. Target
  mới đủ `>1 dB` mở một episode/lịch mới.
- Decision dùng subtype `EarlyTimingDecisionPoint(trigger="early_window")`;
  không giả mạo completed A3+TTT.
- Adapter là opt-in. B0/B0A không thay đổi. RLM, HO preparation/execution,
  RLF recovery và rate accounting dùng lại engine vật lý hiện hành.
- Trong B2/B3, A3/TTT không điều khiển action; vì thế TTT counters bằng 0 và
  không được so trực tiếp với B0/B1. Shadow-A3 cho Rpost/Rearly chưa triển khai.

Tests xác nhận entry strict `>1 dB`, boundary 1 dB không mở, B2 immediate,
B3 thực thi đúng tick +4, hủy khi window đóng, đổi target tạo episode mới,
và không phát decision mới trong HO preparation/execution.

## Dữ liệu

Đúng 8 TRAIN route và 48.000 tick của B0/B0A/B1. Route/file hashes được kiểm
tra qua manifest. Không đọc validation/test/zero-shot, không train model.

## Kết quả aggregate

| Policy | RR % | HOF % | PP % | RLF-time % | HO attempt |
|---|---:|---:|---:|---:|---:|
| B0 | 90,497 | 10,671 | 43,686 | 1,458 | 328 |
| B1 | 90,504 | 10,030 | 43,919 | 1,375 | 329 |
| B2 | 91,246 | 9,420 | 34,400 | 1,083 | 276 |
| B3 | 90,150 | 13,462 | 31,556 | 1,458 | 260 |

So với B0:

- B2: RR `+0,749` điểm %, HOF `-1,250` điểm %, PP `-9,286` điểm %,
  RLF-time `-0,375` điểm %.
- B3: RR `-0,347` điểm %, HOF `+2,791` điểm %, PP `-12,130` điểm %,
  RLF-time không đổi.

B2 point estimate aggregate đạt các mức thực dụng được đăng ký: RR tăng trên
0,5 điểm %, HOF giảm khoảng 11,7% tương đối, PP giảm khoảng 21,3% tương đối và
RLF-time không xấu. Đây **không phải gate PASS**, vì chưa có bootstrap CI và
HOF point estimate theo speed chưa đồng nhất.

## Theo tốc độ

| Speed | B2 ΔRR | B2 ΔHOF | B2 ΔPP | B2 ΔRLF | B3 ΔRR | B3 ΔHOF |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | +0,58 | +1,79 | -9,12 | 0,00 | -0,33 | +4,31 |
| 50 | +0,74 | +0,24 | -19,31 | -0,33 | -0,23 | +0,08 |
| 70 | +1,54 | -2,99 | -8,45 | -0,67 | -0,05 | +6,24 |
| 90 | +0,13 | -2,81 | -2,20 | -0,50 | -0,78 | +1,34 |

Đơn vị các delta là điểm phần trăm. B2 tăng RR ở cả bốn speed theo trung bình
hai route, nhưng HOF xấu nhẹ ở 30/50 km/h. Ở cấp route, `90:70` có RR B2 giảm
0,35 điểm % và thêm hai HOF; `90:108` lại giảm năm HOF nhưng tăng ba PP.
Kết quả này cho thấy hiệu quả phụ thuộc state/route, đúng lý do cần học timing.

## Accounting early action

- B2: 276 entry, 276 HO ngay, delay0; không có lịch cần hủy.
- B3: 267 entry/lịch, 260 HO ở đúng delay4 tick, 7 lịch bị hủy; trong đó ghi
  nhận 4 target changes.

Việc B3 có ít HO và PP hơn không tự động là tốt: nó mất RR và tăng HOF. Điều
này củng cố yêu cầu đánh giá data/risk đồng thời thay vì thưởng giảm HO/PP đơn lẻ.

## Quyết định tiếp theo

Giữ B2 và B3 làm controls. B2 là tín hiệu đáng mở rộng kiểm chứng, nhưng chưa
train PPO. Bước kế tiếp là:

1. bổ sung shadow A3 maturity và regression B0A trong cùng adapter framework;
2. triển khai Rpost/Rearly finite-lookahead cùng target rule;
3. pilot correctness/cost rồi mới chạy 20 TRAIN route/speed với paired
   bootstrap CI;
4. nếu B2 vẫn gần Rearly, ưu tiên rule đơn giản; chỉ mở learning khi adaptive
   timing vượt B2 trên cùng route và vẫn giữ reliability.

Tái lập vào output mới:

```powershell
.\scripts\run_early_ho_b2_b3_pilot.ps1 `
  -OutputDir results/analysis/early_ho_timing/b2_b3_controls_pilot_v2
```
