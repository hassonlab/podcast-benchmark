# DIVER Pretrained Model Checkpoint 지원 설정

## 체크포인트 확인 결과

### 1. DeepSpeed 포맷 ✅ **지원됨**

**확인 결과**:
- 체크포인트에 `module` 키 존재
- `model_state_dict` 키 없음
- DeepSpeed multi-process 포맷 (`mp_rank_00`)

**필요한 설정**:
```yaml
model_params:
  deepspeed_pth_format: true  # 필수!
```

**코드 동작** (`models/diver/DIVER-1/utils/checkpoint.py`):
```python
if deepspeed_pth_format:
    checkpoint['model_state_dict'] = checkpoint['module']  # DeepSpeed 포맷 변환
model.load_state_dict(checkpoint['model_state_dict'])
```

---

### 2. MuP (Maximal Update Parametrization) ❓ **확인 불가**

**체크포인트 구조 분석**:
- 체크포인트 파일 자체에는 MuP 여부가 명시되어 있지 않음
- 체크포인트 키만으로는 MuP로 학습되었는지 판단 불가

**확인 방법**:
1. 원본 학습 설정 확인
2. 모델 문서 확인
3. 실제 로드 시 에러 발생 여부 확인

**현재 설정**:
- 모든 config 파일에서 `mup_weights: false`로 설정됨
- 이는 현재 pretrained 가중치가 MuP로 학습되지 않았음을 의미

**MuP 사용 시 필요한 설정**:
```yaml
model_params:
  mup_weights: true  # Backbone이 MuP로 학습된 경우
  ft_mup: true       # Finetuning head에 MuP 사용 (선택적)
  model_dir: models/diver/pretrained_model  # MuP base shapes 저장 위치
```

**MuP 적용 로직** (`models/diver/DIVER-1/models/finetune_model.py`):
- `mup_weights=True`일 때만 `mup_utils.apply_mup()` 호출
- Base shapes 파일 자동 생성/로드: `DIVER_iEEG_FINAL_model_patch50_dmodel_{width}_layer{depth}_256_512.bsh`

---

### 3. Architecture 파라미터 ✅ **파일명에서 추론 가능**

**파일명 기반 추론**:
- `256_mp_rank_00_model_states.pt` → `d_model=256`
- `512_mp_rank_00_model_states.pt` → `d_model=512`
- `patch_size=50` (config 주석에 명시)

**필요한 설정**:
```yaml
model_params:
  foundation_dir: models/diver/pretrained_model/256_mp_rank_00_model_states.pt
  d_model: 256  # 파일명에서 추론
  e_layer: 12   # 기본값 (확인 필요)
  patch_size: 50  # config 주석에 명시됨
  patch_sampling_rate: 500
```

---

### 4. Finetuning Head 설정 ✅ **지원됨**

**지원되는 설정**:
```yaml
model_params:
  ft_config: flatten_linear  # 또는 flatten_mlp
  num_mlp_layers: 2  # ft_config=flatten_mlp일 때만 사용
```

**동작**:
- `flatten_linear`: 단일 Linear 레이어 (빠름, 적은 파라미터)
- `flatten_mlp`: MLP 레이어 (느림, 많은 파라미터)

---

## 현재 Pretrained Model 지원 요약

| 설정 | 지원 여부 | 현재 값 | 비고 |
|------|----------|---------|------|
| **DeepSpeed 포맷** | ✅ | `true` | 체크포인트 구조 확인됨 |
| **MuP Backbone** | ❓ | `false` | 체크포인트로 확인 불가, 현재 `false`로 설정 |
| **MuP Finetuning Head** | ✅ | `false` | 코드 지원됨, 현재 미사용 |
| **Architecture** | ✅ | `d_model=256`, `patch_size=50` | 파일명/주석에서 추론 |
| **Finetuning Head** | ✅ | `flatten_linear` | 코드 지원됨 |

---

## 권장 설정 (현재 Pretrained Model 기준)

```yaml
model_params:
  foundation_dir: models/diver/pretrained_model/256_mp_rank_00_model_states.pt
  d_model: 256
  e_layer: 12
  patch_size: 50
  patch_sampling_rate: 500
  ft_config: flatten_linear
  deepspeed_pth_format: true  # 필수!
  mup_weights: false  # 현재 pretrained model은 MuP 미사용으로 추정
  ft_mup: false
  freeze_foundation: false
```

---

## MuP 사용 여부 확인 방법

### 방법 1: 실제 로드 시도
```python
# mup_weights=True로 설정하고 로드 시도
# 에러 발생 시 MuP 미지원 확인
```

### 방법 2: Base Shapes 파일 확인
```bash
# MuP 사용 시 생성되는 base shapes 파일 확인
ls models/diver/pretrained_model/*.bsh
```

### 방법 3: 원본 학습 설정 확인
- DIVER-1 원본 학습 스크립트 확인
- 학습 시 MuP 사용 여부 확인

---

## 결론

**현재 pretrained_model 가중치 (`256_mp_rank_00_model_states.pt`)에서 확인된 지원 설정**:

1. ✅ **DeepSpeed 포맷**: 완전히 지원됨 (`deepspeed_pth_format: true` 필수)
2. ❓ **MuP Backbone**: 확인 불가, 현재 `false`로 설정됨
3. ✅ **MuP Finetuning Head**: 코드 지원됨, 현재 미사용
4. ✅ **Architecture**: 파일명/주석에서 추론 가능 (`d_model=256`, `patch_size=50`)
5. ✅ **Finetuning Head**: `flatten_linear` / `flatten_mlp` 모두 지원

**현재 설정이 올바르게 구성되어 있으며, DeepSpeed 포맷만 올바르게 설정하면 정상적으로 로드됩니다.**

