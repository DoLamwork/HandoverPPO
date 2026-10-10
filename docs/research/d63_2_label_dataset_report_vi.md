# D63-2 — bộ nhãn timing và target từ counterfactual TRAIN

Ngày tạo: 06/10/2026. Trạng thái: **hoàn tất bộ nhãn và kiểm tra correctness;
chưa fit probe, chưa train PPO mới**.

## Mục đích

Bộ nhãn này trả lời ba câu hỏi riêng trên cùng state D63-1:

1. **Timing:** nếu vẫn chọn C1, HO ngay hay chờ 100/200/300 ms tốt hơn?
2. **Target:** nếu HO ngay, C1/C2/C3 đem lại gain và risk nào?
3. **Joint:** trong 12 tổ hợp timing × target, từng phương án đổi delivered data
   và rủi ro protocol ra sao?

Không ép mỗi state thành đúng một nhãn `best action`. Mỗi phương án giữ riêng
delta delivered Mbit, relative rate, connected/recovery/interruption time, số HO,
HOF, RLF và ping-pong. Nhãn `strict_useful` chỉ là một view: delivered Mbit tăng,
không metric risk nào xấu hơn control immediate-C1, và action yêu cầu thực sự
được thực thi. Các trade-off vẫn được giữ trong `pareto_class`.

## Dữ liệu và chống leakage

- Chỉ dùng bank TRAIN `d63_timing_target_main_local_v1`; không đọc validation,
  test hoặc zero-shot.
- 1.199 state, 80 route, nguồn state: Always-C1 399, D61 400, D62 400.
- Horizon chính 1,0 s; 14.388 outcome đầu vào = 1.199 × 12 nhánh.
- Bank thực tế dùng tối đa **5 anchor/source/route**. Đây là ít hơn quota tối đa
  10 trong plan nhưng là dữ liệu thật, không lặp state và không chạy lại chỉ để đủ quota.
- Split seed 29 theo `(speed, trajectory_id)`, stratify theo speed: 48 route fit,
  16 tuning, 16 holdout; ở mỗi speed là 12/4/4. Mọi source, anchor và action
  của cùng route luôn nằm trong cùng fold.
- Input được phép cho probe sau này chỉ gồm observation hiện tại, planned delay
  và planned slot. Source, speed, route ID, outcome tương lai và physical target
  tương lai không được dùng làm feature.

## Kết quả nhãn

| Family | Rows | Strict-useful rows | Route có positive | Fit / tuning / holdout positives |
|---|---:|---:|---:|---:|
| Timing | 4.796 | 278 | 69/80 | 182 / 48 / 48 |
| Target | 3.597 | 77 | 42/80 | 46 / 17 / 14 |
| Joint | 14.388 | 747 | 76/80 | 470 / 160 / 117 |

Target-only là bài toán thưa nhất nhưng toàn bank có 77 positive, vượt mốc
adequacy sơ bộ 50 positive của plan. Holdout vẫn chỉ có 14 target positive trên
9 route, vì vậy probe target phải báo uncertainty theo route và không được chỉ
dựa vào accuracy/AP tổng.

Một số nhánh delay không kịp thực thi trước horizon hoặc rơi vào fallback được
gắn `unavailable_or_fallback`, không được xem là candidate yêu cầu đã thành công.
Điều này đặc biệt quan trọng với timing/joint và ngăn nhãn C2/C3 hưởng outcome
của fallback C1.

## Artifact tái lập

- `results/analysis/d63_2_labels_v1/input_registry.json`: hash đầu vào và hợp đồng feature.
- `route_split_manifest.json`: mapping route → fold và kiểm tra leakage.
- `timing_labels.npz`, `target_labels.npz`, `joint_labels.npz`: observation,
  planned action, continuous gain/risk và các nhãn teacher.
- `label_report.json`: counts tổng và theo fold.
- `label_manifest.json`: kích thước và SHA-256 của gói.
- Script tái lập: `scripts/prepare_d63_2_labels.ps1`; script từ chối ghi đè output cũ.

## Gate tiếp theo

Bước kế tiếp hợp lệ là fit probe D63-2 theo đúng thứ tự: fit model trên `fit`,
chọn threshold trên `tuning`, chỉ sau khi khóa mới mở `holdout`. Chạy riêng
timing, target và joint; báo AP/prevalence, calibration, net physical gain,
false-alternative loss và bootstrap theo route. Chưa có kết quả probe nên chưa
được kết luận observation hiện tại học được opportunity hoặc mở pilot PPO mới.
