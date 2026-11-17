# Early-HO T1 — đối chiếu B0 và B0A

Ngày: 10/10/2026. Nhánh: `codex/early-ho-timing`.
Artifact chính: `results/analysis/early_ho_timing/b0_b0a_pilot_v2`.
Trạng thái gate: **PASS**.

## Mục tiêu

Kiểm tra adapter action-controlled hiện tại có tái hiện đúng baseline 3GPP khi
policy luôn chấp nhận HO tại completed A3+TTT hay không:

- B0: `A3ReferenceSimulator.predict` tự bắt đầu HO khi TTT hoàn tất.
- B0A: `HandoverEngine` dừng tại completed A3+TTT, sau đó
  `AlwaysHandoverPolicy` gọi `start_handover()` tới đúng target đã latch.

B0A trong phép kiểm tra này **chưa mở early window và chưa HO trước A3+TTT**.

## Dữ liệu và protocol

- Chỉ dùng TRAIN, split 70/15/15, split seed 17.
- Sample seed 17, 2 route/tốc độ 30/50/70/90 km/h, tổng 8 route.
- Route: `30:62`, `30:170`, `50:20`, `50:218`, `70:124`, `70:142`,
  `90:70`, `90:108`.
- Mỗi route 6.000 sample, dt 10 ms; tổng 48.000 primitive tick.
- Runtime giữ nguyên 2,1 GHz, 10 MHz, A3 offset 1 dB, hysteresis 1 dB,
  TTT 40 ms và toàn bộ RLM/HO/recovery mặc định.
- Không đọc validation, test hoặc zero-shot; không train model.

Manifest lưu SHA-256 của 16 file RSRP/SINR, dataset provenance, năm source
file quyết định phép audit, Git commit, trạng thái dirty và hash tracked diff.
Output tồn tại sẽ bị từ chối ghi đè.

## Đối chiếu

Yêu cầu equality tuyệt đối `rtol=0`, `atol=0`, với NaN phải ở cùng vị trí:

- từng tick: serving-cell timeline, HO-execution flag, RLF-recovery flag,
  serving RSRP và serving SINR;
- pre/post-HO SINR, successful HO history và ping-pong indices;
- mọi history và final state của N310, N311, T310, RLF recovery, TTT,
  HO preparation và HO execution;
- toàn bộ metric do `get_statistics()` trả về.

Kết quả: **8/8 route PASS, mismatch = 0**.

| Metric chung của B0 và B0A | Giá trị |
|---|---:|
| Mean route relative rate | 90,497% |
| HO attempts / successes / failures | 328 / 293 / 35 |
| HOF | 10,671% |
| Ping-pong | 43,686% |
| RLF recovery time | 1,458% |

Các số trên chỉ mô tả 8 TRAIN route pilot, không phải kết quả benchmark cuối
cùng. Ý nghĩa của PASS là mọi số của B0A đúng bằng B0 trên cùng route.

## Kết luận được phép

B0A là adapter tương đương quan sát với B0 trên pilot: việc tách thời điểm
completed A3+TTT thành decision rồi chấp nhận ngay không tự làm thay đổi
protocol, target, QoS hay HOF/PP/RLF. Vì vậy B0A có thể làm control nội bộ khi
xây early-decision adapter.

Không được suy ra từ phép kiểm tra này rằng:

- HO sớm tăng QoS hoặc giảm HOF/PP;
- B0/B0A là tối ưu;
- parity đã được chứng minh cho toàn bộ 1.060 route;
- một policy WAIT/HO khác cũng giữ parity.

## Bước tiếp theo

Giữ test B0/B0A làm regression bắt buộc. Tiếp theo mới xây early-window
adapter opt-in và kiểm thử boundary/target-change/WAIT20. Sau đó chạy pilot
B1/B2/B3/Rpost/Rearly; chưa mở main audit hoặc train PPO trước khi correctness
và cost gate T1 hoàn tất.

Chạy tái lập vào một output mới:

```powershell
.\scripts\run_early_ho_b0_b0a_parity.ps1 `
  -OutputDir results/analysis/early_ho_timing/b0_b0a_pilot_v3
```
