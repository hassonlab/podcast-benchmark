# MuP 사용 시 주의사항

## ⚠️ 중요: MuP는 임의로 켜면 안 됩니다!

### 1. `mup_weights` (Backbone MuP) - ❌ 임의로 켜면 안 됨

**문제점**:
- `mup_weights=True`는 **pretrained 가중치가 MuP로 학습되었을 때만** 사용해야 함
- MuP로 학습되지 않은 가중치를 MuP 모드로 로드하면:
  - 가중치의 파라미터화 방식이 맞지 않음
  - 모델 성능이 크게 저하될 수 있음
  - 학습이 불안정해질 수 있음

**코드 동작** (`models/diver/DIVER-1/models/finetune_model.py`):
```python
# 1. Backbone을 mup=True로 생성
self.backbone = DIVER(d_model=width, e_layer=depth, mup=mup, patch_size=patch_size)

# 2. mup_weights=True이면 MuP base shapes 적용
if params.mup_weights:
    mup_utils.apply_mup(target_module=self.backbone, ...)

# 3. 그 후 가중치 로드
self.load_backbone_checkpoint(params.foundation_dir, ...)
```

**문제**:
- MuP로 학습되지 않은 가중치를 MuP 모드로 로드하면, 가중치의 스케일과 파라미터화가 맞지 않음
- MuP는 파라미터 초기화와 학습 방식이 다르므로, 일반 방식으로 학습된 가중치는 MuP 모드에서 제대로 작동하지 않음

**올바른 사용**:
```yaml
# ✅ Pretrained 가중치가 MuP로 학습된 경우만
model_params:
  mup_weights: true  # 원본 학습 시 MuP 사용 여부와 일치해야 함
```

---

### 2. `ft_mup` (Finetuning Head MuP) - ⚠️ 주의 필요

**상황**:
- Finetuning head는 새로 학습하는 부분이므로, `ft_mup=True`로 설정하는 것이 기술적으로는 가능함
- 하지만 **권장되지 않음**:
  - Backbone과 finetuning head의 파라미터화 방식이 일관되지 않음
  - 학습 안정성 문제 가능

**코드 동작** (`models/diver/DIVER-1/models/finetune_model.py`):
```python
# flatten_linear_finetune에서
use_mup = params.ft_mup
self.ft_core_model = MakeModelIgnoreDataInfoList(
    nn.Linear(in_dim, out_dim) if not use_mup
    else MuReadout(in_dim, out_dim, output_mult=1.0)  # MuP 사용
)
```

**권장 사용**:
```yaml
# ✅ Backbone이 MuP로 학습된 경우에만 finetuning head도 MuP 사용
model_params:
  mup_weights: true   # Backbone이 MuP로 학습됨
  ft_mup: true        # Finetuning head도 MuP 사용 (일관성 유지)
```

```yaml
# ✅ Backbone이 일반 방식으로 학습된 경우
model_params:
  mup_weights: false  # Backbone이 일반 방식으로 학습됨
  ft_mup: false       # Finetuning head도 일반 Linear 사용 (일관성 유지)
```

---

## 현재 Pretrained Model 상황

### 확인된 사실:
- 체크포인트 파일명: `256_mp_rank_00_model_states.pt`
- DeepSpeed 포맷: ✅ 확인됨
- MuP 여부: ❓ 체크포인트로 확인 불가

### 현재 설정:
```yaml
model_params:
  mup_weights: false  # 현재 설정 (올바름)
  ft_mup: false       # 현재 설정 (올바름)
```

### MuP 사용 여부 확인 방법:

#### 방법 1: 원본 학습 설정 확인
- DIVER-1 원본 학습 스크립트 확인
- 학습 시 `mup=True`로 설정되었는지 확인

#### 방법 2: Base Shapes 파일 확인
```bash
# MuP 사용 시 생성되는 base shapes 파일 확인
ls models/diver/pretrained_model/*.bsh

# 파일이 있으면: MuP 사용 가능
# 파일이 없으면: MuP 미사용 또는 아직 생성 안 됨
```

#### 방법 3: 실제 테스트
```yaml
# 테스트용 설정 (주의: 성능 저하 가능)
model_params:
  mup_weights: true
  # ... 기타 설정
```
- 실제로 로드하고 학습해보기
- 성능이 크게 저하되면 MuP 미지원 확인

---

## 결론

### ❌ 하지 말아야 할 것:
```yaml
# MuP로 학습되지 않은 가중치에 MuP 사용
model_params:
  mup_weights: true  # ❌ 위험! 원본 학습 방식과 일치하지 않음
```

### ✅ 올바른 사용:
```yaml
# 1. Pretrained 가중치가 MuP로 학습된 경우
model_params:
  mup_weights: true   # ✅ 원본 학습과 일치
  ft_mup: true        # ✅ (선택적) 일관성 유지

# 2. Pretrained 가중치가 일반 방식으로 학습된 경우 (현재 상황)
model_params:
  mup_weights: false  # ✅ 원본 학습과 일치
  ft_mup: false       # ✅ 일관성 유지
```

### 현재 권장 설정:
```yaml
model_params:
  foundation_dir: models/diver/pretrained_model/256_mp_rank_00_model_states.pt
  deepspeed_pth_format: true
  mup_weights: false  # ✅ 현재 설정 유지 (원본 학습 방식 확인 전까지)
  ft_mup: false       # ✅ 현재 설정 유지
```

**MuP를 사용하려면 먼저 pretrained 가중치가 MuP로 학습되었는지 확인해야 합니다!**

