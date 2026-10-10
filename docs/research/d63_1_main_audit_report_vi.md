# D63-1 — báo cáo main audit, kiểm chứng và tái lập

Ngày: 06/10/2026. Trạng thái: **PASS**.

## Phạm vi đã chốt

- Chỉ dùng TRAIN: 20 route/tốc độ, tổng 80 route.
- State bank có 1.199 anchor; tối đa **5 anchor/source/route**.
- Kế hoạch cho phép tối đa 10 anchor/source/route. Gói thực tế dùng 5; đây là
  quota nhỏ hơn đã ghi nhận và không cần chạy lại chỉ để lấp quota.
- Full-route có 320 hàng = 80 route × 4 policy.
- Bootstrap ghép cặp theo route, phân tầng speed: 2.000 lần, seed 29.
- Không chạy training mới, validation, test hay zero-shot.

## Định nghĩa metric đã kiểm tra

- Relative rate từng route = `100 × estimated_rate / max_estimated_rate`;
  aggregate là trung bình relative rate theo route.
- Mean rate là trung bình `estimated_rate_mbps` theo route.
- HOF% = `100 × tổng HOF / tổng HO attempt`.
- RLF-time% = `100 × tổng recovery seconds / tổng thời gian route`.
- Ping-pong% = `100 × tổng ping-pong / tổng HO success`.
- QoS retention dùng tỷ lệ giữa các aggregate mean và được kiểm tra riêng theo
  từng tốc độ.

## Kết quả full-route TRAIN

| Policy | RR (%) | HOF (%) | RLF-time (%) | PP (%) |
|---|---:|---:|---:|---:|
| Always-C1 | 89,868 | 11,800 | 1,576 | 36,065 |
| D61 frozen | 85,639 | 4,882 | 0,555 | 13,729 |
| Timing-only reference | 91,294 | 3,965 | 0,321 | 19,464 |
| Timing×target reference | 91,606 | 4,431 | 0,338 | 19,405 |

So với Always-C1:

- Timing-only tăng RR `+1,426` điểm phần trăm, CI95%
  `[1,267; 1,584]`.
- Timing×target tăng RR `+1,738` điểm phần trăm, CI95%
  `[1,569; 1,907]`.
- QoS retention theo từng speed đều lớn hơn 100% đối với cả hai reference.
- HOF, RLF-time và ping-pong đều giảm theo point estimate và bootstrap interval
  tương ứng không cắt qua hướng xấu.

Gate G1: **PASS**. Kết quả chứng minh có dư địa timing và target trên TRAIN với
finite lookahead 1 giây. Reference dùng future để tạo chuẩn chẩn đoán, chưa phải
policy có thể triển khai online và chưa đánh giá G2.

## Kiểm chứng và provenance

Script tái lập:

```powershell
.\scripts\finalize_d63_1.ps1
```

Script thực hiện `py_compile`, chạy 20 test D63/protocol, kiểm tra:

- đủ 320 full-route rows và cùng 80 route cho bốn policy;
- 20 route ở mỗi tốc độ 30/50/70/90 km/h;
- công thức RR/HOF/RLF-time/PP từ raw denominators;
- accounting `decision = STAY + HANDOVER`;
- SHA-256 của 7 input artifact, 160 file RSRP/SINR và checkpoint D61/D62;
- TRAIN-only, không có training mới và G1 trong các artifact khớp nhau.

Output kiểm chứng mặc định:

```text
results/analysis/d63_1_verification_v1/
  verification_report.json
  d63_1_reproducibility_report_vi.md
```

Script từ chối ghi đè output đã tồn tại. Muốn kiểm chứng lại, truyền một
`-OutputDir` mới.

Local bank registry đã lưu runtime source-tree hash. Main full-route run chưa
nhúng source-tree hash ngay lúc bắt đầu; gói kiểm chứng lưu hash code hiện tại
dưới dạng post-run snapshot. Đây là giới hạn provenance được ghi rõ, không được
diễn giải post-run snapshot như bằng chứng mật mã tuyệt đối của code tại thời
điểm chạy.

## Quyết định tiếp theo

D63-1 đã chốt. Bước tiếp theo là D63-2: dùng observation nhân quả hiện tại để
kiểm tra khả năng dự đoán timing trước, sau đó đo giá trị bổ sung của target.
