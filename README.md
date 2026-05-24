# Bài tập nhóm cuối kì: Tối ưu hóa giao hàng đa tác tử thời gian thực

Bài toán mô phỏng hệ thống giao hàng thực tế: một đội `C` shipper hoạt động đồng thời trên bản đồ lưới `N × N`, nhận và giao các kiện hàng có trọng lượng, mức ưu tiên và deadline khác nhau.

Đơn hàng xuất hiện liên tục theo thời gian với tốc độ biến động, bao gồm các đợt cao điểm (_surge_), tập trung tại một số khu vực đặc biệt (_hotspot_), tạo ra nút cổ chai cục bộ.

Nhiệm vụ của nhóm là thiết kế thuật toán phân công và điều phối shipper để tối đa hóa tổng phần thưởng trong `T` bước thời gian, cân bằng giữa:

- Giao đúng hạn
- Xử lý đơn ưu tiên cao
- Chi phí di chuyển

---

## Thang thời gian

- `1 giờ = 10` đơn vị thời gian
- `1 ngày = 240` đơn vị thời gian

---

# Files được cấp

| File                  | Mô tả                                                     |
| --------------------- | --------------------------------------------------------- |
| `run_test.py`         | Grader chính thức, chấm điểm tự động. **Không được sửa.** |
| `test_config.txt`     | 6 config Phase 1 kèm bản đồ. **Không được sửa.**          |
| `demo_notebook.ipynb` | Kaggle notebook gọi terminal, không chứa thuật toán.      |

---

# Phase 1 — Phát triển và nộp bài

Nhóm nhận `test_config.txt` để phát triển và kiểm tra các thuật toán được yêu cầu:

```bash
python run_test.py --config test_config.txt --out results/ --seed 42
```

Code và báo cáo nộp ở Phase 1 là phiên bản chính thức được đánh giá và chấm.

Các nhóm cài đặt các thuật toán trong thư mục `solvers`, upload thư mục lên Kaggle dạng Dataset và để public vào hôm deadline của Phase 1 để giảng viên có thể xem.

Nhóm sẽ không được sửa notebook này, các bạn chỉ được thay đổi dòng copy từ thư mục private của nhóm vào thư mục `solvers` trong Notebook chấm mẫu.

> Nộp trên Kaggle: **1 version duy nhất**, vi phạm sẽ bị trừ điểm.

---

# Phase 2 — Config dùng để Ranking

Ba ngày trước deadline Phase 2, giảng viên sẽ công bố file config test dùng để ranking.

File này sẽ được cập nhật trên chính file config cũ để chấm.

Các notebook đã nộp sẽ không được thay đổi, vi phạm sẽ bị trừ điểm.

---

# 1. Mô tả bài toán

Cho bản đồ dạng lưới `A` kích thước `N × N`, trong đó:

- `A[i][j] = 0`: ô trống
- `A[i][j] = 1`: ô vật cản

với:

```text
1 <= i, j <= N
```

Tại `t = 0`, có `C` shipper trên bản đồ.

Shipper `i` có:

- Tọa độ `(x_i, y_i)`
- Tải trọng tối đa `W_max(i)`
- Sức chứa `K(i)` đơn

Không có hai shipper nào đứng cùng ô.

---

# 1.1. Tập hành động

Tại mỗi bước `t`, mỗi shipper thực hiện một cặp hành động:

```text
(move, cargo_op)
```

## Move

```text
move ∈ {S, L, R, U, D}
```

| Ký hiệu | Ý nghĩa        |
| ------- | -------------- |
| `S`     | Đứng yên       |
| `L`     | Di chuyển Tây  |
| `R`     | Di chuyển Đông |
| `U`     | Di chuyển Bắc  |
| `D`     | Di chuyển Nam  |

## Cargo Operation

| Giá trị  | Ý nghĩa                 |
| -------- | ----------------------- |
| `0`      | Không làm gì            |
| `1`      | Nhặt đơn tại ô hiện tại |
| `2 [id]` | Giao đơn `id` đang mang |

## Thứ tự trong một bước

```text
Di chuyển → Nhặt hàng → Giao hàng
```

---

# 1.2. Mô hình đơn hàng

Một đơn hàng `g_i` được biểu diễn bởi:

```text
g_i = <sx_i, sy_i, ex_i, ey_i, et_i, w_i, p_i>
```

| Thuộc tính      | Ý nghĩa               |
| --------------- | --------------------- |
| `sx_i, sy_i`    | Tọa độ điểm lấy hàng  |
| `ex_i, ey_i`    | Tọa độ điểm giao hàng |
| `et_i`          | Deadline              |
| `w_i`           | Khối lượng kiện hàng  |
| `p_i ∈ {1,2,3}` | Mức ưu tiên           |

## Mức ưu tiên

| Giá trị | Ý nghĩa    |
| ------- | ---------- |
| `1`     | Tiêu chuẩn |
| `2`     | Nhanh      |
| `3`     | Hỏa tốc    |

---

# 1.3. Mô hình sinh đơn hàng: Surge & Hotspot

Đơn hàng xuất hiện theo quá trình Poisson không đồng nhất:

```text
lambda(t)
```

## Trong surge window

Nếu:

```text
t ∈ [t_s, t_e]
```

thì:

```text
lambda(t) = lambda_0 × (1 + A)
```

Ngược lại:

```text
lambda(t) = lambda_0
```

## Tham số

| Tham số            | Ý nghĩa              |
| ------------------ | -------------------- |
| `lambda_0 ≈ G / T` | Tốc độ sinh đơn nền  |
| `A >= 0`           | Biên độ surge        |
| `[t_s, t_e]`       | Surge window         |
| `Hotspot (r, c)`   | Tâm khu vực đặc biệt |

---

## Cơ chế hotspot

Trong surge window:

- `70%` xác suất:
  - Điểm lấy hàng được chọn trong vùng Manhattan `<= 3` quanh hotspot
- `30%` còn lại:
  - Chọn ngẫu nhiên toàn bản đồ

Điều này tạo ra nút cổ chai cục bộ:
nhiều đơn xuất hiện gần nhau trong thời gian ngắn.

---

## Ví dụ trực quan

### Bình thường (`lambda_0 = 0.1`)

```text
. . . . .
. . o . .
. o . . .
. . . o .
. . . . .
```

### Trong surge (`A = 3.0`, `lambda = 0.4`)

```text
. . . . .
. H H H .
. H * H .
. H H H .
. . . . .
```

---

## Phase 1

Các tham số:

- `lambda_0`
- `A`
- danh sách surge windows
- hotspots

**không được công bố** nhằm khuyến khích thiết kế thuật toán thích nghi môi trường động.

Nếu config không có các trường này thì hệ thống sẽ random theo cấu hình code.

---

## Phase 2

Tham số surge và hotspot được công bố đầy đủ.

---

# 1.4. Sức chứa và trọng lượng

Mỗi shipper `i` phải thỏa mãn đồng thời:

## Ràng buộc tải trọng

```text
sum(w_j for j in bag(i)) <= W_max(i)
```

## Ràng buộc số lượng đơn

```text
|bag(i)| <= K(i)
```

---

## Ưu tiên khi nhặt hàng

Nếu một ô có nhiều đơn:

```text
Hỏa tốc > Nhanh > Tiêu chuẩn > Chỉ số nhỏ hơn
```

---

## Bảng chi phí

| Hạng mục   | Khối lượng        | Chi phí/bước | Sức chứa |
| ---------- | ----------------- | ------------ | -------- |
| Nhẹ        | `w <= 3 kg`       | `-0.01`      | `3 đơn`  |
| Trung bình | `3 < w <= 10 kg`  | `-0.02`      | `2 đơn`  |
| Nặng       | `10 < w <= 30 kg` | `-0.04`      | `1 đơn`  |
| Siêu nặng  | `w > 30 kg`       | `-0.08`      | `1 đơn`  |

---

# 1.5. Hàm phần thưởng

## Giao đúng hạn

Nếu:

```text
t_delivery <= et_i
```

thì:

```text
r(i) = alpha_p × r_base(i) × (1 + bonus)
```

---

## Giao trễ

Nếu:

```text
t_delivery > et_i
```

thì:

```text
r(i) = beta_p × r_base(i) × max(0, 1 - (t_delivery - et_i) / T)
```

---

## Bonus

```text
bonus = max(0, (et_i - t_delivery) / et_i)
```

---

## Hệ số ưu tiên

| Loại dịch vụ | p   | alpha_p | beta_p |
| ------------ | --- | ------- | ------ |
| Hỏa tốc      | 3   | 3.0     | 0.5    |
| Nhanh        | 2   | 2.0     | 0.3    |
| Tiêu chuẩn   | 1   | 1.0     | 0.1    |

---

## Phần thưởng cơ bản

```text
r_base(i) = 10 × f_weight
```

| Trọng lượng       | f_weight | r_base |
| ----------------- | -------- | ------ |
| `w <= 0.2 kg`     | 0.4      | 4      |
| `0.2 < w <= 3 kg` | 1.0      | 10     |
| `3 < w <= 10 kg`  | 1.5      | 15     |
| `10 < w <= 30 kg` | 2.0      | 20     |
| `w > 30 kg`       | 3.0      | 30     |

---

# 1.6. Chi phí di chuyển

Chi phí di chuyển của shipper `i` tại bước `t`:

```text
rc(i, t) = -0.01 × (1 + gamma × W_carried(i, t) / W_max(i))
```

với:

```text
gamma = 1.0
```

Chỉ áp dụng khi shipper di chuyển:

```text
L, R, U, D
```

Đứng yên `S` không mất chi phí.

---

# 1.7. Hàm mục tiêu

Mục tiêu:

```text
maximize sum_i [
    sum reward của các đơn giao bởi shipper i
    + sum_t rc(i, t)
]
```

Tức là:

- tối đa hóa phần thưởng giao hàng
- đồng thời tối ưu chi phí di chuyển

---

# 1.8. Các ràng buộc vận hành

## Va chạm

Shipper có chỉ số nhỏ hơn được ưu tiên giữ ô khi tranh chấp.

## Thứ tự thao tác

```text
Di chuyển → Nhặt hàng → Giao hàng
```

## Giới hạn di chuyển

- Không được ra ngoài bản đồ
- Không được vào ô vật cản

## Ràng buộc luôn đúng

```text
W_max(i) và K(i)
```

phải luôn được thỏa mãn.

---

# 2. Các phương pháp cần cài đặt

## Yêu cầu với mỗi phương pháp

- Trình bày độ phức tạp thời gian và không gian
- Phân tích mức độ tối ưu:
  - optimal
  - near-optimal
  - heuristic
- So sánh kết quả định lượng trên config Phase 1

---

# Bắt buộc — 5 điểm/phương pháp

1. Greedy BFS
2. VRP + OR-Tools

---

# Nâng cao — 2.5 điểm/phương pháp

1. Ant Colony Optimization (ACO)
2. Multi-Agent Pickup and Delivery với Conflict-Based Search (MAPD-CBS)

---

# 5. Cách nộp bài

```text
submission/
├── solvers/
├── run_test.py
├── test_config.txt
├── demo_notebook.ipynb
└── report.pdf
```

## Mô tả

| File/Folder           | Ý nghĩa                  |
| --------------------- | ------------------------ |
| `solvers/`            | Code thuật toán của nhóm |
| `run_test.py`         | KHÔNG ĐƯỢC SỬA           |
| `test_config.txt`     | KHÔNG ĐƯỢC SỬA           |
| `demo_notebook.ipynb` | Notebook Kaggle submit   |
| `report.pdf`          | Báo cáo kỹ thuật         |

---

# Quy tắc Kaggle notebook

- Share đúng `1 version`
- Notebook chỉ chạy terminal (`%%bash`)
- Không chứa thuật toán
- Seed cố định:

```bash
--seed 42
```

---

# 6. Thang điểm

| Hạng mục         | Điểm   | Mô tả                        |
| ---------------- | ------ | ---------------------------- |
| Greedy BFS       | 5      | Chạy đúng trên tất cả config |
| VRP + OR-Tools   | 5      | Chạy đúng trên tất cả config |
| ACO              | 2.5    | Tốt hơn Greedy BFS           |
| MAPD-CBS         | 2.5    | Xử lý xung đột đa tác tử     |
| Báo cáo kỹ thuật | 5      | Theo yêu cầu                 |
| Ranking Phase 2  | 10     | Theo net reward              |
| Vấn đáp          | 20     | Trình bày & trả lời          |
| **Tổng**         | **50** |                              |

---

# 6.1. Yêu cầu báo cáo kỹ thuật

## Nội dung bắt buộc

- Danh sách thành viên (tối đa 3)
- Phân công công việc
- Mô tả từng thuật toán
- Độ phức tạp thời gian/không gian
- Mức độ tối ưu
- Bảng so sánh kết quả:
  - net reward
  - % đơn đúng hạn
  - thời gian chạy
- Phân tích trade-off

---

## Nâng cao

Mô tả chiến lược xử lý:

- surge
- hotspot
- phát hiện cao điểm
- điều phối tài nguyên

---

# 6.2. Điểm ranking

Điểm ranking dựa trên:

```text
Tổng net reward của run_test.py với config Phase 2
```

## Quy tắc tính điểm

- Nhóm cao nhất: `10 điểm`
- Các nhóm còn lại:
  - tính theo tỉ lệ tuyến tính

---

## Điều kiện

Notebook phải:

- chạy độc lập
- hoàn thành trong `60 phút`

---

## Lưu ý quan trọng

Trong khi chạy lại:

- Giảng viên sẽ cập nhật trực tiếp `test_config.txt`
- Nhóm chỉ được run lại notebook
- Không được chỉnh sửa notebook

---

# Hạn chế

- Thời gian chạy tối đa: `60 phút`
- Không được dùng internet
- Không được cài thêm thư viện
- `pip install` sẽ không hoạt động

Nếu thư mục `solvers` chứa file không phải code:

- phải mô tả cách tạo file
- nộp kèm quy trình sinh file

---

# Chú ý

Nếu ở Phase 2:

- vi phạm điều kiện
- hoặc code không chạy được

thì:

```text
Điểm ranking = 0/10
```

Nhóm có thể:

- chọn phương pháp tốt nhất để nộp
- hoặc chương trình tự lấy phương pháp có kết quả tốt nhất
