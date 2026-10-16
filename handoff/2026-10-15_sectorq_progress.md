# SectorQ 진행 상태 — 2026-10-15 00:50 UTC

이 파일은 `2026-10-14_sectorq_handoff.md`의 후속 기록이다. 원래 0절 규칙을 유지한다. 모든 쓰기는 kcj 안에서, 우리 `pp_` job만, 새 8노드 이상 job은 사용자 승인 뒤, 할당 하나에 loop 하나. push는 요청이 있을 때만 한다.

## 이번 턴의 결과

- 인계된 변경을 먼저 커밋했다: `15b6a6f`(승인된 설계 그림/절 구성), `0c6f3bc`(고정 행 실험 재개기).
- 설계 3.1–3.5, 확보된 결과를 사용한 평가, 관련 연구 본문을 작성했다. 설계 개념 그림은 수정하지 않았다.
- 고정 행 비교, 1–512 GPU scale-out, 원래 회로 기준 128 GPU 비교, 128–512 GPU 시간 분해 그림을 추가하고 weak 결과를 갱신했다.
- 서론 수정 초안을 먼저 제시한 뒤 인용, ORQA 분배 함수 설명, GPU 구현에 대한 주장 범위, 메모리 산식을 정정했다. 사용자가 승인한 `In this paper` 문단은 바이트 단위 비교로 유지됨을 확인했다.
- 인용 TODO와 서지정보의 미확인 TODO를 제거했다. Atlas, Häner–Steiger, tapering, commutator graph, iQCC, paulistrings-rs, ORQA, MonoProp, Julia, SPD, Qiskit, cuPauliProp, PADO-Pauli, Backpropagating PP의 원문/공식 저장소를 확인했다.
- `bash paper/scripts/build.sh` 성공, `python3 paper/scripts/check_style.py` 0건. 26쪽, 검사기의 본문 문단 합계 9,467단어(초록 169, 승인 서론 마지막 문단 190, 결론 106). 미해결 인용/참조와 Overfull 없음. acmart의 기존 그림 description/익명 소속 경고와 일부 Underfull, 학회 서지의 address 경고는 남는다.
- PDF: `/pscratch/sd/s/sgkim/kcj/pauliprop/paper/main.pdf`. GitHub 링크는 https://github.com/changjongkim/SectorQ/blob/main/paper/main.pdf 이며 **push하지 않았으므로 이번 수정은 아직 없다**.

## 실험 상태와 재개

### N2 균형 선택 ablation — 실행 전

`jobs/b_ablate.sh`를 만들었다. 기본 4노드/16 GPU, 같은 `pp_engine_x`, 8개 후보 대 1개 후보, trace 기록, 일곱 회로다. `--cands 1`은 후보 간 비교만 없애며, 표본 수집과 sector 수 선택까지 없애는 것은 아니다.

한 차례 제출이 `QOSMaxSubmitJobPerUserLimit`로 거절되었고 job은 만들어지지 않았다. `logs/b/loop.log`에 오류가 있다. 기다리는 loop는 없다. 공유 계정의 다른 작업을 중지하거나 재사용하지 않았다.

재개 전 `logs/b/loop.pid`, 해당 `/proc/<pid>/cmdline`, `timeout 15 pgrep -u sgkim -f 'jobs/b_ablate.sh'`, 할당의 active step을 확인한다. 오래된 PID가 다른 프로세스에 재사용되었으면 그 프로세스를 건드리지 않는다. 실행기는 campaign flock과 allocation flock을 사용하며 요청은 `--immediate=30`으로 한 번만 한다. 재시도 대기 loop는 없다. 이후 코드에는 종료 시 직접 만든 `pp_b` 할당만 반납하는 trap도 추가했다.

다음 턴에 interactive 제출 여유가 있으면 중복 프로세스가 없음을 확인한 뒤 다음과 같이 분리 실행한다. 루트 build.sh와 대기 job의 바이너리는 다시 빌드하지 않는다.

```python
import subprocess
with open('logs/b/loop.log', 'ab', buffering=0) as log:
    subprocess.Popen(['nohup', 'setsid', 'bash', 'jobs/b_ablate.sh'],
                     stdin=subprocess.DEVNULL, stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
```

`PP_B_JOB`는 idle 상태의 4노드 `pp_` 할당에만 사용한다. 다른 campaign loop가 그 할당을 다시 사용할 가능성이 없어야 한다. 실행 중 기다리지 말고 확보된 결과를 보고한다.

결과가 생기면 wall time, 기대값, GPU/host split, 노출된 row wait, 구간별 가장 큰 sector/평균을 비교한다. trace의 `sectors`는 **비어 있지 않은 sector 수**이다. 평균은 nonempty 기준과 전체 `2**rows` 기준을 구별한다. `analysis/paper_ablation.py`의 `parse_log()`가 두 불균형을 계산한다. N2 효과가 작거나 음수이면 먼저 사용자에게 알리고 독립적인 성능 주장을 낮춘다.

### 대기 중인 job — 변동 없음

00:50 기준 모두 PENDING(Priority):

| Job | Name | Nodes |
|---|---|---:|
| 59768854 | pp_s_g256w | 64 |
| 59768995 | pp_s_g32w | 8 |
| 59768996 | pp_s_g64w | 16 |
| 59768998 | pp_s_g128w | 32 |
| 59768999 | pp_s_g256w2 | 64 |
| 59769000 | pp_s_g512w | 128 |
| 59769001 | pp_p_g32 | 8 |
| 59769002 | pp_p_g64 | 16 |

남은 반영 대상은 N2, CPU 추가 ladder/weak/reach, paulistrings-rs 32/64 GPU, 재측정 중인 256 GPU 한계다. 평가 파일의 `% TBD` 주석 다섯 곳에 위치를 남겼다. 원고와 실험 전체가 완료되었다고 보고하면 안 된다.

## 새로 확인한 근거와 주의점

1. `queue_s/todo_a0o_orig.jsonl`의 원래 회로 cuPauliProp 66건은 인계문과 달리 이미 모두 terminal 상태다. 성공 41, OOM 14, 실패 이후 생략 11. 128 GPU 원래 회로 KI/noise 결과를 사용할 수 있어 새 그림에 적용했다. 128 GPU 최대 공통 완료 설정에서 SectorQ 7.4–13.4배. TFIM은 원래 입력과 frame 입력이 같다.
2. 고정 행 ablation은 `logs/x/*.out` 중 정확한 `re|fx.<circuit>.g4|g16.<cutoff>.out`만 사용한다. 하위 contended/old_buggy/alloc_end, `.t1` 등은 제외했다. 성공 쌍은 16 GPU 25개, 4 GPU 22개이며 출력 기대값은 모두 같다. 서로 window 경계가 다르므로 **최대 boundary 항 수는 두 모드에서 같을 필요가 없다**. 그림 x 항 수는 recomputed 모드 기준이다.
3. 16 GPU 고정/재계산 시간 비율 2.43–17.03. KI 1e-5는 137.069/11.938초, 878/20구간, 698.507/45.221 GB 전송. 기존 element sequence의 SectorQ 12.2초와 별도 할당이므로 하나로 합치지 않았다.
4. 기본 gmax=8은 512 GPU에서도 boundary sector 상한 256이다. KI 3e-6은 trace에서 최대 256개 nonempty sector와 split 0, 256→512 GPU wall 10.478→9.669초, idle worker 비중 9.0→40.2%. TFIM/noise에서도 유사하다. 이 설정의 효율 저하를 네트워크 하나로 설명하면 안 된다.
5. 상한만 늘리면 해결된다고도 쓰지 않는다. 기존 `n12.ki.3e-6.g512`는 gmax=12와 chunk=65536을 함께 사용했고 11.568초로 기본 9.669초보다 느리다. 이는 두 파라미터의 공동 변경이라 cap 단독 ablation은 아니다.
6. 최대 512 GPU 실행 `i12.ki.1.5e-7.g512`는 3,413.465초, boundary 448.2B, 17.9 TB payload, host chunk pool 21,079 GiB(설치 host의 64%). library operations 46%, exchange 26%, GPU-host copies 약 9%, idle 0.5%. exchange 중 transfer 607초, posting 280초, counts 2초 미만. 잔여 시간을 전부 planning으로 부르지 않는다.
7. worker의 operations/idle 등은 **누적 worker 시간**이며 GPU utilization 샘플이 아니다. worker 수로 나눈 평균과 노드 평균 exchange/sync를 wall과 비교했다. `analysis/paper_large_time.py` 참조.
8. `max_terms_inside_a_window`는 각 sector 실행의 최대 항 수를 더한 값이다. 동시에 메모리에 존재하는 전역 최대라고 해석하면 안 된다. 서론은 실제 boundary 159M(1e-5), 11.856B(1e-6), 40 B/term으로 약 6.4/475 GB를 사용한다. 배경은 내부 합산 counter의 정의를 명시했다. 기존 `4 TB = GPU 1,000장` 문장은 제거했다.
9. KI 낮은 cutoff의 기대값은 단조 수렴하지 않는다(1e-6부터 약 0.3217, 0.3256, 0.3265, 0.3226, 0.3153, 0.3089). 정확한 물리 오차가 반드시 작아졌다는 주장과, 동일 truncation의 더 큰 실행을 가능하게 했다는 주장을 구분했다.

## 재생성 및 검증

```bash
python3 analysis/paper_ablation.py
python3 analysis/paper_large_time.py
cd figures
../env/bin/python3 make_paper_figures.py ev_fixed_rows ev_large_time ev_scaleout ev_circuits128 ev_weak
cd ..
bash paper/scripts/build.sh
python3 paper/scripts/check_style.py
```

원본 로그는 기존 경로에 유지했다. 분석 JSON/CSV에는 상대 로그 경로가 있다. 고정 행 JSON은 summary만 저장하며 trace가 필요하면 parse_log()로 원본을 읽는다. 큰 규모 JSON에는 trace도 있다. 추가 그림만 생성하고 설계 그림 생성기는 실행하지 않는다.

## 사용자의 novelty 질문에 대한 답변 방향

현재 확인한 선행 연구 기준으로 N1이 시스템 기여의 중심이다. parity/선형 분할 자체, stage 실행 자체, host offload 자체가 새롭다고 주장하지 않는다. 구간의 모든 생성자로부터 merge-local partition을 구성하고, 살아 있는 항에 맞춰 재선택하며, 분할 가능한 sector를 host memory에서 GPU로 실행하는 메커니즘이 핵심이다. 고정 행 ablation은 N1의 실질적 효과를 뒷받침한다. N2의 후보 선택 효과는 실험 전이므로 독립적으로 강하다고 단정하지 않았다. 관련 연구 본문은 이 구분에 맞췄다.

## Git 및 미완료 의사결정

코드와 논문 커밋을 분리하고 경로를 명시했다. 논문 커밋에는 Co-Authored-By를 넣지 않는다. push하지 않는다. 다른 작업의 staged `queue_f/ext.log`, `queue_f/runner.log`, untracked `jobs/e14_scaleout.sh`, compare 그림과 로그는 손대지 않았다.

공개 저장소 익명화와 승인된 서론 마지막의 anonymous URL 유효성은 아직 사용자 결정/확인이 필요하다. 새 8노드 이상 실험이 필요해지면 결과와 구체적 계획을 준비한 뒤 승인받는다.

## 후속 변경 — Evaluation 통합 (2026-10-15)

- 사용자의 후속 요청으로 `af8e176`까지 GitHub main에 push를 완료했다. 위의 push 전 상태 설명은 00:50 시점 기록이다.
- 이후 사용자가 Evaluation의 분리된 결과를 합칠 수 있는지 점검해 달라고 요청했다. 기존 11개 소절, 15개 그림, 24개 제목 문단을 6개 소절, 12개 그림, 19개 제목 문단으로 재구성했다.
- 새 순서는 Setup → Performance Compared with Existing Systems → Scalability and Capacity → Contribution of the Design Choices → Numerical Agreement and Accuracy → Parameter Sensitivity and Repeatability다.
- 성능 비교에 single-node, 16 GPU, 128 GPU, paulistrings-rs를 모았다. paulistrings-rs는 truncation 주기가 다르므로 cuPauliProp 비교 곡선과 합치지 않았다.
- 항당 시간 7개 패널은 실행 시간/항 수의 재표현이므로 본문에서 중복 그림을 제거하고 지표 정의와 수치를 16 GPU 성능 문단에 통합했다. 원본 ev_cost PDF와 생성 코드는 저장소에 그대로 있다.
- Strong/weak scaling, 회로 크기/깊이, 최대 operator/host 사용량을 한 소절로 모았다. 회로 크기/깊이는 한 문단, 최대 항 수/host 사용량도 한 문단으로 통합했다.
- 고정 행 ablation과 요소별 계단, 시간 분해를 같은 소절에 배치했다. KI의 구간 수/전송량 중복 설명을 제거했다. N2가 완료되면 이 소절 안에 추가하며 새 독립 소절을 만들지 않는다.
- 16 GPU 및 128–512 GPU 시간 분해는 한 그림의 6개 패널로 합쳤다. 동일한 색상 범주를 확인하고 기존 large-time 범례를 공통으로 썼다. 퍼센트/초 축의 차이는 캡션에 명시했다.
- Parameter sensitivity와 반복 측정은 한 소절 및 한 그림으로 묶되, 실험 조건과 범례를 각 패널 묶음에 유지했다. 구현 간 답 일치는 Accuracy로 옮겼다.
- 새 데이터/설계 그림은 그리지 않았다. 기존 그림 PDF 83개가 모두 수정 전 SHA-256과 같음을 확인했다. 합친 그림을 PDF 19/21쪽에서 렌더링해 확인했다.
- 결과: Evaluation 본문 3,601 → 2,757단어, 전체 본문 9,467 → 8,623단어, 원고 26 → 24쪽. 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. 대기 실험의 TBD 주석 다섯 곳은 유지했다.
- 이 구조 변경은 로컬 커밋으로 남긴다. 앞선 push 요청은 이전 버전에 대해 완료했으며 이번 구조 변경은 아직 push하지 않았다.

## 후속 변경 — Figure 17의 2×3 배치 (2026-10-15)

- 사용자의 추가 push 요청으로 Evaluation 통합 커밋 `9c7af6b`를 GitHub main에 push했다.
- 사용자가 Figure 16 유지, Figure 17만 수정으로 명확히 지정했다. 앞서 작업 중이던 Figure 16 삭제/표 압축과 Background 수정은 되돌렸다. Figure 16과 Table 6, Background는 `9c7af6b`와 같다.
- Figure 17의 (a)–(c)를 가로 누적 막대에서 세로 누적 막대로 바꾸고, 아래 128–512 GPU 패널과 같은 크기·색상·글자 크기를 사용했다. 위 3개, 아래 3개와 공통 범례로 배치했다.
- 위쪽은 16 GPU의 기존 다섯 cutoff별 시간 비중(%), 아래쪽은 기존 128/256/512 GPU 실행 시간(s)이다. 기존 `g_breakdown_sizes`가 반환하는 동일 측정값을 재사용했다. 새 실험이나 데이터 변경은 없다.
- `figures/make_paper_figures.py ev_breakdown`만 실행했다. 데이터 패널 3개와 해당 범례만 갱신했고, 설계 그림과 나머지 그림 PDF는 SHA-256으로 동일함을 확인했다.
- 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. PDF 19쪽을 렌더링해 2×3 배치와 다섯 cutoff의 가독성을 확인했다. 본문 8,623단어(Evaluation 2,757단어)는 그대로이며 원고는 24 → 23쪽이다.
- 이번 Figure 17 변경은 로컬 커밋만 한다. 새로운 push 요청은 아직 없다. 실험 대기 상태는 이 편집 작업에서 다시 조회하지 않았다.

## 후속 변경 — 데이터 그림의 폭과 글자 정리 (2026-10-15)

- Figure 17 수정 후 사용자의 push 요청에 따라 `9060b77`까지 GitHub main에 push했다.
- 사용자가 Figure 9·11의 가로폭/여백 개선을 요청했고, 이어서 모든 데이터 그림의 축 외 굵은 글씨 제거, 작은 폰트, 데이터 글자 넘침 해결로 범위를 넓혔다. 설계 그림은 그대로 유지했다.
- Figure 9·11은 4+3 배치를 유지하면서 행마다 가로폭을 채운다. 같은 행의 실제 플롯 폭을 맞추고, 동일한 로그 y축의 눈금/제목은 행의 첫 패널에만 둔다. 옅은 가로 격자와 일관된 배수 표시를 사용한다. Figure 11의 범례에도 배수의 정의를 넣었다.
- 모든 데이터 패널/범례를 재생성했다. 축 제목만 bold, 눈금과 범례는 11 pt, 데이터 표시는 10 pt, 축 제목은 11.5 pt다. 본문 삽입 배율 0.6 기준 6.6/6.0/6.9 pt로, 기존 눈금·범례 12 pt와 수치 11 pt보다 약 8–9% 작다.
- 데이터 글자는 기준점을 유지하고 텍스트 위치를 조정해 플롯 내부에 둔다. 축 모서리의 눈금 충돌은 간격을 확보해 해결한다. 긴 Background y축 제목은 두 줄로 나누고, 해당 설명과 메모리 민감도 끝점 숫자 위에 여백을 확보했다. Figure 12의 범례도 공통 0.6 배율을 사용한다.
- 원본 측정값·분석 파일은 수정하지 않았다. 데이터 PDF 78개만 갱신되었고, 설계/연산 그림 5개는 수정 전 SHA-256과 같다. 별도로 보관하던 항당 시간 그림 생성은 `ev_cost`로 분리했으며 전체 생성 시에는 계속 포함된다.
- 생성기의 글자/범례/데이터 경계 및 겹침 검사 0건. 데이터 그림이 있는 PDF 4–6, 12–21쪽을 모두 렌더링해 확인했다. 컴파일 성공, check_style 0건, 미해결 인용/참조 및 Overfull 없음. 본문 8,623단어(Evaluation 2,757단어), 23쪽이다.
- 재생성은 `figures/`에서 `MPLCONFIGDIR=/pscratch/sd/s/sgkim/kcj/pauliprop/paper/build/matplotlib ../env/bin/python3 make_paper_figures.py`로 한다. 검사 로그와 미리보기는 git 제외인 `paper/build/`에 있다.
- 이번 변경은 로컬 커밋만 하며 새 push 요청은 아직 없다. 이 편집 작업에서는 실험/job 상태를 다시 조회하지 않았다.

## 후속 변경 — Figure 17의 cutoff별 3×3 배치 (2026-10-15)

- 앞선 데이터 그림 정리 후 사용자의 push 요청으로 `6fcb59c`까지 GitHub main에 push했다.
- 사용자가 Figure 17도 cutoff별로 보여 달라고 요청했고, GPU 수는 16·64·256의 세 행으로 지정했다. 열은 kicked Ising, TFIM heavy-hex, noisy kicked Ising이다. 모든 패널을 시간 비중(%)과 공통 로그 cutoff 축으로 통일했다.
- 새 `analysis/paper_time_sweeps.py`가 기존 완료 로그 103개를 JSON/CSV로 정리한다. 16 GPU는 기존 그림과 동일한 15개 실행, 64/128/256/512 GPU는 기본 cap=256, truncEvery=10, 회로별 동일 target의 88개 실행이다. 그림에는 16/64/256 GPU의 59개만 사용하고 128/512 GPU도 분석 파일에 보관했다. 새 실험은 없다.
- 16 GPU의 1e-4/3e-5/1e-5는 `logs/g/eng.*`의 `pp_engine_h`, 3e-6/1e-6은 `logs/r/eng.*`의 `pp_engine_i`로 기존 출처를 유지했다. 64 GPU 이상은 `logs/s/i.ref.ki.*`와 `logs/s/n.eng.*`의 `pp_engine_i`다. 같은 바이너리 버전으로 모두 재측정한 sweep이라고 주장하지 않는다.
- higher-cap/chunk KI 실행은 비교 조건이 다르므로 새 cutoff 그림에서 제외했다. 기존 `paper_large_time.*`와 최대 용량 결과는 그대로다. worker counter/GPU 수와 node-average exchange/sync를 합산하며, Other는 wall에서 남는 시간이다. 103개 모두 음수 성분이 없고 합이 wall과 일치한다.
- 본문에 64 GPU의 idle worker 비중이 cutoff 1e-4에서 41–48%, 회로별 최저 표시 cutoff에서 1–2%라는 결과를 추가했다. 기존 512 GPU의 idle 40%는 별도 측정임을 명시했다.
- `figures/make_paper_figures.py ev_runtime`으로 새 패널 9개와 공통 범례만 생성했다. 기존 PDF 83개는 SHA-256이 모두 같아 Figure 16과 설계 그림을 포함한 다른 그림은 바뀌지 않았다.
- 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. 19쪽을 렌더링해 배치와 가독성을 확인했다. 23쪽, 본문 8,647단어(Evaluation 2,781단어)다. 미리보기는 `paper/build/runtime-cutoffs.png`에 있다.
- 새 변경은 코드/논문 커밋을 분리해 로컬에 남긴다. 이번 변경에 대한 새 push 요청은 아직 없다. 이 편집 작업에서는 실험/job 상태를 다시 조회하지 않았다.

## 후속 변경 — 대표 cutoff, 축 표기, 요구사항 표 제거 (2026-10-15)

- 앞선 Figure 17 변경은 사용자 요청으로 `daecc96`까지 GitHub main에 push했다. 아래 변경에 대한 새 push 요청은 아직 없다.
- 사용자가 Figure 17을 cutoff 1e-4/1e-5/1e-6만 보이도록 줄이고, 빠진 결과가 대기 실험으로 채워지면 16/64/256 GPU 세 행을 유지하도록 요청했다. 세 회로의 세 점이 현재 엔진/기본 상한으로 모두 완료된 규모는 16 GPU다. 64/128/256/512 GPU의 TFIM/noise는 세 점이 있으나 KI의 1e-6은 다른 엔진 또는 상한을 바꾼 실행만 있었다.
- 기존 대기 목록에는 KI 1e-6의 engine i / cap 256 측정이 없었다. 기존 `pp_s_g64w`(59768996, 16노드), `pp_s_g256w2`(59768999, 64노드)의 script가 시작 시 `queue_s/todo*.jsonl`을 읽는 것을 확인했다. 두 job은 확인 당시 PENDING(Priority), 3시간 할당이었다. 기존 pp_ job 8개도 모두 PENDING이었다.
- `queue_s/todo_a00_runtime.jsonl`에 `i.ref.ki.1e-6.g64`와 `i.ref.ki.1e-6.g256` 두 항목을 추가했다. bin/pp_engine_i, target 6M, truncEvery 10, 기본 gmax=8/chunk/window, trace 1, vram fraction 0.9다. 예상 600/240초, 각 1,200초 제한이며 기존 목록 앞쪽에서 실행한다. 새 Slurm job/loop를 만들지 않았고 실행 중인 바이너리를 빌드하지 않았다. 각 ID의 done/log/lock이 없는지와 기존 실행기의 항목/바이너리 선택을 검증했다. 256 GPU의 다른 기존 할당이 먼저 claim해도 기존 lock으로 중복 실행을 막는다.
- Figure 17은 세 cutoff의 25개 완료 값과 KI 2개 Pending이다. 미측정을 0으로 그리지 않는다. 16 GPU도 `logs/r/eng.*`의 재측정으로 통일해 전체 시간 분석 자료 103개가 모두 pp_engine_i, cap 256이다. 분석기는 두 보충 로그가 생기면 각각 독립적으로 읽을 수 있다. higher-cap/chunk 자료는 기존 별도 분석에 남긴다.
- Evaluation의 일부 눈금이 비었던 원인은 겹침 방지용 생략 코드였고, 괄호는 그 코드의 축약 단계에서 제거되거나 fixed-row 생성기에서 처음부터 빠졌다. 처음에는 모든 눈금을 엇갈려 표시했으나, 사용자가 긴 라벨의 중간 생략을 허용하고 cutoff 축의 여러 줄 배치를 원하지 않는다고 명확히 수정했다. 최종 규칙은 cutoff와 `(항 수)` 또는 `(오차)`를 한 줄에 유지하고, 겹치면 중간 눈금만 생략하며 양 끝은 남기는 것이다. Figure 17은 cutoff 세 개만 한 줄로 표시한다. 그림의 측정점 자체는 생략하지 않는다.
- Figure 9/11은 4+3 배치와 행별 동일 플롯 크기를 유지한다. Figure 16의 항 수 괄호를 복원했다. Figure 15에는 Execution Configuration 축 제목을 넣고, 범례의 규모를 intermediate 합산값(22.5M/1.4B)에서 본문 정의에 맞는 boundary 항 수(2.33M/159M)로 바로잡았다. Accuracy의 괄호는 항 수가 아니라 오차라는 구분을 유지한다.
- 사용자가 기존 Table 4(Requirements of distributed Pauli propagation)의 역할을 지적했다. R1–R6은 후속 Design/Evaluation에서 쓰이지 않고 Background를 반복하며, R6의 작은 문제에서 손실이 없어야 한다는 요구는 결과보다 강했다. 해당 표와 참조를 삭제하고 Background의 locality/adaptation/capacity 관찰을 Design의 세 메커니즘으로 잇는 두 문장으로 대체했다. Evaluation 순서를 요구사항 번호에 맞춰 재편하지 않았다. 이제 Runtime parameters가 Table 4, fixed-row window/traffic이 Table 5다.
- 최종 검증: 컴파일 성공, check_style 0건, 미해결 참조/인용과 Overfull 없음. 23쪽, 본문 8,674단어(Evaluation 2,804단어). Evaluation PDF 64개를 갱신했고 Background와 설계 그림은 모두 기존 SHA-256과 같다. 최종 6쪽 및 12–21쪽을 렌더링해 확인했다. 최종 패널 생성의 겹침/넘침 검사도 0건이다. 미리보기는 `paper/build/cutoff-final-*.png`와 `table-removal-final.png`에 있다. 보충 측정 두 건은 아직 미완료이며 기다리지 않고 현재 원고를 저장한다.

## 후속 변경 — Figure 17의 GPU 규모별 문제 크기 (2026-10-15)

- 사용자가 Figure 17의 왼쪽 1e-4는 유지하고 64/256 GPU에는 더 큰 항 수의 문제를 사용하도록 요청했다. 완료 데이터만 쓰는 선택지와 목표 규모의 미완료 점을 대기로 표시하는 선택지를 제시했으며, 사용자는 후자를 명시적으로 선택했다.
- 세 회로 모두 cutoff를 16 GPU에서 1e-4/1e-5/1e-6, 64 GPU에서 1e-4/5e-6/5e-7, 256 GPU에서 1e-4/2e-6/3e-7로 고정했다. idle 비중으로 고르지 않는다. 가운데 점의 boundary terms/GPU는 각 회로에서 규모 간 1.5배 이내다. 오른쪽은 더 큰 용량을 평가하며, 고정 입력의 strong scaling과 구분한다. 기존 default-cap 512 GPU의 idle/동시성 한계 설명은 유지했다.
- 22개 점은 완료, 5개 점은 Pending이다. 64 GPU KI의 오른쪽 점은 `n10.ki.5e-7.g64`(44.299B boundary terms, 1,590.499 s, 평균 worker idle 0.7%)다. 256 GPU KI는 기존 `r12.ki.3e-7.g256` 재측정 대기를 사용한다. 오래된 engine h의 117.465B 완료 시간은 새 그림에 섞지 않았다.
- 빠진 TFIM/noisy KI 네 점을 `queue_s/todo_a0_capacity_runtime.jsonl`에 넣었다. `i.runtime10.{tfim,noise}.5e-7.g64`, `i.runtime12.{tfim,noise}.3e-7.g256`, binary는 모두 기존 `pp_engine_i`다. 각각 기존 16노드/64노드 pp_s job의 같은 loop가 읽는다. TFIM 제한/예상은 3,600/2,400초, noise는 1,800/900초다. 기존 256 GPU KI 재측정 목록 다음, 나머지 capacity 목록 앞에 온다. 새 Slurm job/loop나 바이너리 빌드는 없다. 중복 ID, bin, done/log/lock 부재, runner 지원 및 다섯 Pending의 실제 queue 항목을 검증했다. 전에 넣은 KI 1e-6 default-cap 두 측정도 통제 비교용으로 유지했다.
- 큰 오른쪽 점만 64 GPU에서 cap 1,024/chunk 262,144, 256 GPU에서 cap 4,096/chunk 65,536을 사용한다. 나머지는 기본 cap 256/chunk 262,144다. 별표와 본문에 이를 명시했다. 모든 데이터는 engine i이며 target/truncation/circuit 설정을 검사한다.
- 새 `analysis/paper_runtime_profiles.py`가 고정한 선택 규칙과 provenance를 `paper_runtime_profiles.json/csv`에 저장한다. 미완료 항 수/시간은 null이며, 실패 시 Pending으로 숨기지 않고 Failed/Incomplete로 표시한다. 기존 `paper_time_sweeps.*`는 통제된 default-cap 자료로 그대로 남겼다. 완료 뒤에는 `python3 analysis/paper_runtime_profiles.py`, `figures/`에서 `MPLCONFIGDIR=../paper/build/matplotlib ../env/bin/python3 make_paper_figures.py ev_runtime` 순서로 갱신하고 본문의 Pending 개수/해석도 함께 고친다.
- x축은 한 줄로 세 cutoff를 모두 표시하고 양 끝의 완료 항 수만 괄호로 붙인다. 긴 가운데 항 수만 생략해 cutoff 자체가 사라지지 않게 했고 세 눈금 유지 assertion도 넣었다. 미완료 항 수를 이전 엔진이나 추정치로 채우지 않는다.
- Figure 17의 아홉 패널과 범례만 갱신했고, 나머지 83개 그림 PDF는 기존 SHA-256과 같다. 컴파일 성공, check_style 0건, 미해결 인용/참조 및 Overfull 없음. 23쪽, 본문 8,742단어(Evaluation 2,872단어). PDF 18/19쪽을 시각 확인했고 plot 검사도 0건이다. 미리보기는 `paper/build/runtime-capacity-final-01.png`, `-02.png`다.
- 관련 기존 job은 확인 당시 `pp_s_g64w`(59768996), `pp_s_g256w`(59768854), `pp_s_g256w2`(59768999)가 모두 PENDING(Priority)였다. 기다리지 않는다. 이번 변경 및 앞선 `24491d2`, `e54efa6`은 새 push 요청이 없어 로컬에만 저장한다. GitHub main은 `daecc96`이다.

## 후속 변경 — 항 수를 cutoff 아래에 정렬 (2026-10-15)

- 위의 변경은 후속 사용자 요청으로 `09036bf`까지 GitHub main에 push했다. 이후 사용자가 모든 데이터 그림의 항 수를 cutoff 바로 아래 줄로 내려 정렬하도록 명확히 정정했다. **현재 유효한 지침은 cutoff 윗줄 / `(항 수)` 아랫줄이며, 앞선 한 줄 표기 지침을 대체한다.** 긴 축은 왼쪽·가운데·오른쪽 세 지점의 라벨을 표시한다.
- 공통 생성기에서 숫자와 괄호 안 수치를 두 줄로 묶고, 두 줄 모두 같은 눈금의 중심에 정렬한다. 표시할 가운데는 양 끝의 중간 위치에 가장 가까운 실제 측정 눈금이다. 표시된 세 쌍은 모두 유지하며, 선/막대의 측정점 자체는 생략하지 않는다. 짧은 K/M/B/T 항 수 표시는 두 유효숫자로 반올림하지만 원본 분석 데이터와 측정값은 바꾸지 않았다.
- Figure 17은 완료된 가운데 점의 항 수를 복원했다. 세 cutoff 모두 윗줄에 맞추고, Pending의 아랫줄은 비워 높이를 맞춘다. 미측정 항 수를 추정해 넣지 않는다. cap 변경의 별표와 22개 완료/5개 대기 구성은 그대로다.
- Figure 9/11은 4+3 배치와 같은 행의 동일 플롯 폭을 유지하면서 두 줄 눈금을 위한 하단 공간을 확보했다. 네 점짜리 좁은 패널의 좌우 여백을 조정해 세 라벨을 유지한다. 데이터 패널의 글자/범례는 normal, 축 제목만 bold인 규칙은 그대로다. Accuracy의 괄호 값은 여전히 오차이며 항 수로 바꾸지 않았다.
- Background 데이터와 보관 중인 데이터 패널까지 모두 갱신했다. Figure 1은 원래의 설계 그림+데이터 두 패널을 한 행에 유지했다. 좁은 KI 패널에서 첫 수치가 선/다른 수치에 겹치지 않게 위치를 조정했다. 원고에서 사용하지 않는 항당 시간 패널도 같은 두 줄 형식을 적용하기 위해 3열 패널 폭을 사용한다.
- 데이터 PDF 88개를 재생성했다. `overview`, `ops`, `resector`, `execution`, `sectors`의 다섯 PDF는 수정 전 SHA-256과 같다. 모든 생성 함수의 최종 실행에서 겹침/넘침 검사 0건이며, 선택된 세 눈금과 두 줄 형식·괄호를 assertion으로 확인한다. 시각 확인은 데이터 그림이 있는 4–6쪽과 12–21쪽을 대상으로 했고, Figure 1의 최종 수정은 `paper/build/terms-bg-final.png`에서 확인했다.
- 최종 컴파일 성공, check_style 0건, 미해결 참조/인용과 Overfull 없음. **23쪽, 본문 8,742단어(Evaluation 2,872단어)**로 유지된다. 최종 PDF는 `paper/main.pdf`다. 이 변경은 로컬 커밋만 하며 새 push 요청은 아직 없다. job/실험 목록은 이번 라벨 수정에서 변경하지 않았다.

## 후속 변경 — 원고 문체와 논증 구조 정리 (2026-10-15)

- 항 수의 두 줄 정렬 변경은 사용자 요청으로 `3484a69`까지 push했으며, `git push origin main`과 원격 SHA 대조로 일치를 확인했다. 아래 문체 수정은 그 뒤의 별도 로컬 변경이다.
- 활성 목표는 “미사여구·구어체·비유적 표현을 제거하고 ScaleQsim 논문 구조와 프로젝트 문장 규칙에 맞춰 수정”이다. ScaleQsim의 실제 절 구성과 CASCADE의 문단 구조, 원래 handoff와 논문 메모리의 규칙을 다시 확인했다. 이번 목표 완료는 문체·구성 수정의 완료이며 대기 실험을 포함한 연구 전체의 완료가 아니다.
- 초록부터 결론까지 7개 절 파일의 문장을 검토했다. 배경의 `truncation must see`, `merge partners meet`, `partitions follow the run`, `wrong answer` 등은 실제 coefficient 합산·partition label·측정 expectation 설명으로 교체했다. 일반적인 장점의 반복이나 novelty 자체를 논평하던 문장은 구체적인 조건과 구현 차이로 바꿨다.
- 서론은 수정 초안을 `paper/build/intro-prose-draft.tex`로 먼저 제시한 뒤 반영했다. ScaleQsim의 응용 맥락 → classical reference 필요 → 표현 방식 → 측정된 제약 → 선행 연구 → 설계 → 기여 순서를 유지했다. 승인된 마지막 `In this paper` 문단은 바이트 단위로 동일하며 190단어다. 제목과 공개 주소도 바꾸지 않았다.
- Background는 3개 소절, Design은 N1/N2/N3 중심의 기존 5개 소절을 유지했다. 서론과 중복되던 표현 방식 소개는 Background 도입에서 Pauli propagation의 정의로 정리했다. fixed-row 구조 분석의 Clifford frame 없는 조건을 본문에 명시했다. Design에는 측정 결과를 넣지 않았고 파라미터·4-qubit 예시·단계 설명만 유지했다. row selection의 overlap이 항상 완전히 숨겨진다는 표현을 제거하고 worker 대기 조건을 설명했다.
- Evaluation은 기존 6개 소절을 유지했다. 두 문단으로 되어 있던 시간 분석의 둘째 문단에 `Concurrency and Exchange Costs.` 제목을 부여했다. 수치와 비교 조건, 기준선의 작은 문제 우위, 다른 truncation rule, CPU reproduction, cap/chunk 동시 변경, 비단조 오차, Pending 및 TBD를 유지했다. 서로 다른 cutoff에서 계산한 항당 시간은 matched-size speedup이 아님을 명확히 했다. 시간 분석의 평균화와 Other 정의도 본문에 넣었다.
- Related Work는 ScaleQsim처럼 주제별 두 소절로 정리했다. 첫 소절은 conserved subspaces/fixed partitions/staged state vectors, 둘째는 distributed CPU/libraries/repeated evaluation and gradients다. 전체 인용 key 집합은 수정 전과 동일하다.
- 캡션은 명사구로 정리했다. Table 1 약어는 표 아래 설명으로, Figure 17의 별표·괄호·미완료 설명은 인접 본문으로 옮겼다. 설계 execution 그림의 chunk 색/빗금 설명은 캡션에 유지했다.
- 최종 검증: `bash paper/scripts/build.sh` 성공, `python3 paper/scripts/check_style.py` 0건, `git diff --check` 통과, 미해결 인용·참조/Overfull/TeX 오류 없음. 기존 acmart의 익명 소속과 그림 description 경고는 남는다. `paper/main.pdf`와 `paper/build/main.pdf`는 동일하다. **23쪽, 본문 8,315단어, Evaluation 2,910단어, 초록 142단어·5문장, 결론 93단어·3문장**이다.
- 정규식 검사 외에 문단 도입/끝의 논리, Design 숫자의 파라미터/예시 여부, 캡션, Evaluation 세미콜론 부재, 동사 do/did 부재, sentence-initial Thus/Therefore를 수동·보조 검사했다. 기존 checker는 `do not`을 허용하는 빈틈이 있으므로 해당 표현도 직접 제거했다. 검사기 코드는 바꾸지 않았다.
- 그림 PDF 93개(설계 5개 + 데이터 88개)는 모두 수정 전 SHA-256과 같다. Figure 16 유지, Figure 17 3×3/22개 완료/5개 Pending, cutoff 윗줄·항 수 아랫줄 정렬은 그대로다. 렌더링한 1/2/4/7/10/12/17/18/19/20/21/23쪽에서 문단과 표·그림 배치를 확인했다. 검토 자료와 snapshot은 무시되는 `paper/build/prose-*`에 있다.
- 상태 조회에서 기존 pp_ job 8개는 모두 PENDING이었다. 새 job/loop 제출, 바이너리 빌드, 실험 목록 변경은 없었다. `queue_f/ext.log`, `queue_f/runner.log`의 다른 작업 staged 변경과 untracked 자료는 보존했다.

## 후속 변경 — 16 GPU N2 및 overlap 실험 실행 (2026-10-16)

- 원고 문체 수정 `73758b9`는 후속 사용자 요청으로 push를 완료했다. 이후 새로 완료된 실험이 있는지 조회했으나 기존 8개 대규모 job은 모두 Priority 대기였고, Figure 17의 큰 문제 5개도 미완료였다. 05:52 UTC의 N2 재시도는 계정의 QOSMaxSubmitJobPerUserLimit에 거절되었다.
- 사용자가 16 GPU에서 추가 실험을 점검하도록 요청했다. N1 고정/재계산은 25개 성공 비교쌍, Figure 17의 16 GPU는 9/9 완료, sector target/per-GPU task 설정은 각각 9개 추가 측정이 이미 있다. N3도 `logs/r/eng.{ki,tfim,noise}.1e-6.g16.out`에서 실제 GPU/host split과 exit=0을 확인했다(KI 12,280/365회, TFIM 9,031/187회, noise 1,300/24회). 따라서 이 증거를 얻기 위한 재실행은 불필요하다. GPU 메모리 제한 sweep 자체는 기존 1/4 GPU 결과이며 16 GPU 결과로 잘못 표기하면 안 된다.
- 남은 16 GPU 실험은 ① 후보 1/8개 비교(7회로 30설정 × 2 = 60회)와 ② overlap on/off(KI/TFIM/noise × cutoff 1e-4/1e-5 × 2 × 3반복 = 36회)다. 사용자가 “다시 돌려야하는것까지 모두 돌려줘 interactive 잡고”로 실행을 명시했다. 현재 loop는 이 96회와 같은 할당에서 다시 측정하는 대조군을 포함한다.
- `jobs/b_ablate.sh`를 확장했다. 동일 pp_engine_x, seed 7, truncation/target/default cap을 유지하고 한 번에 하나의 srun만 실행한다. 앞 42회는 작은 N2 설정, 다음 36회는 overlap, 마지막 18회는 큰 N2 설정이다. overlap의 두 번째 반복은 순서를 반대로 한다. 각 로그에 allocation, 정확한 args, 바이너리 SHA-256, 코드 commit을 기록한다. 이전 로그는 `.previous.<new-job>.<UTC>`로 보존하고 전부 현재 할당에서 다시 측정한다. 실패한 실행은 exit를 남기며 다음 점을 측정 실패로 추정하지 않는다. 각 실행 뒤 남은 step이 있으면 다음 실행 없이 할당을 종료한다.
- `analysis/paper_n2.py`는 해당 할당의 완료/미완료/실패를 구별하고, 같은 할당·바이너리·입력 옵션에서 `--cands` 또는 `--overlap-plan` 하나만 다른지 검사한다. 기대값 차이가 1e-9를 넘으면 invalid이며 시간 비율에서 제외하고 오류를 반환한다. worker 대기는 worker 수로 나누고, trace 불균형은 nonempty/all sector 기준을 따로 유지한다. 기본은 읽기만 하며, `--allocation <job> --write`는 할당별 JSON에 기록한다. loop 종료 시 자동으로 한 번 기록한다.
- 준비 커밋 `60ddc2a`와 수정 커밋 `cbcf9cd`는 로컬이다. bash 문법, 96개 고유 항목/48쌍/반복수 검사, 분석기의 다른 할당·바이너리·seed·기대값 거부 및 실패/미완료 분류 검사를 통과했다. 실행 중 바이너리는 재빌드하지 않았다.
- 첫 할당 `59842646`은 06:14:47 UTC 시작했고 KI 대조군 한 건(exit=0) 뒤 종료됐다. srun이 while-read의 실험 목록 표준입력을 소비한 것이 원인이었다. 이 할당과 loop가 끝났음을 확인한 뒤, srun stdin을 /dev/null로 하고 목록은 전용 fd 7에서 읽도록 수정했다. 원래 한 건은 `.previous.59842747.*`와 `analysis/paper_n2.59842646.json`에 보존되며 현재 비교에 섞지 않는다.
- **현재 할당 `59842747`, 이름 pp_b, 4노드/16 GPU, 4시간, 06:16:27 UTC 시작, loop PID 1872405**. 노드는 nid001212,nid001213,nid001909,nid002064다. 시작 코드 `cbcf9cdd5f6dbda17202c770b0601f06fd602cd1`, pp_engine_x SHA-256 `d426695d7053e6bf629120bb61c1ea0d808e8e771fbdbd80627a2da9f4fe20be`. `/proc/1872405/cmdline`과 pgrep으로 loop 하나를 확인했고 여러 후속 실행까지 정상 진행했다. 실행 목록은 `logs/b/plan.59842747.tsv`, 진행 로그는 `logs/b/loop.log`다. **실행 중 jobs/b_ablate.sh를 편집하거나 같은 할당에 다른 loop를 시작하지 않는다.**
- 중간 확인에서 10회 성공/0회 실패/0회 invalid, 5개 성공 비교쌍의 출력 기대값 차이 0이었다. 작은 cutoff 1e-4에서는 후보 1개가 8개보다 빠른 초기 결과(1/8 시간비 0.659-0.987)가 나왔다. 사용자에게 알렸으며, 첫 실행의 초기화·순서 영향과 단일 측정이라는 한계를 포함해 해석해야 한다. 큰 문제 완료 전 N2의 속도 이득을 단정하지 않는다. 후보 1개도 표본 수집과 sector 수 선택은 유지하므로 N2 전체를 끈 비교가 아니다.
- 상태 확인은 `squeue -j 59842747`, `tail logs/b/loop.log`, `PYTHONDONTWRITEBYTECODE=1 python3 analysis/paper_n2.py --allocation 59842747`. 종료 후 같은 명령에 `--write`를 추가하고 결과를 평가 4.4의 N2 위치와 시간 분석에 반영한다. 원고 수정 시 기존 컴파일/check_style/PDF 링크 규칙을 적용한다. 이번 실행 준비에서는 원고·그림을 바꾸지 않았고 새 커밋은 push하지 않았다. 긴 대기 loop 없이 현재 상태를 보고한다.

## 후속 변경 — 16 GPU ablation 완료와 Figure 18 (2026-10-16)

- 할당 59842747은 06:58:29 UTC에 96회 전부 성공해 종료했다. 30개 후보 비교쌍과 18개 overlap 반복 비교쌍 모두 같은 할당/바이너리/입력 조건 검증을 통과했고 출력 기대값 차이는 0이다. loop가 할당을 반납했으므로 sacct의 CANCELLED 표시는 측정 실패가 아니다. 이전의 현재 실행 중 표시는 이 항목으로 대체한다.
- 후보 8개는 30조건 중 28조건에서 최대 nonempty-sector 불균형을 줄였다(최대값들의 중앙값 1.069 → 1.029). 후보 1개/8개의 시간비 중앙값은 0.9967, 전체 범위는 0.6594–1.0571이다. 각 조건은 seed 7의 단일 측정이므로 작은 시간 차이를 일반적인 가속으로 주장하지 않는다. 후보 1개도 표본 수집과 sector 수 결정을 유지한다.
- Overlap은 KI/TFIM/noisy KI × cutoff 1e-4/1e-5의 6조건에 각 3회 반복이다. 비활성/활성 평균 시간비는 1.030–1.466(6조건 기하평균 1.215)이다. 96회 데이터와 원고 단락은 로컬 커밋 9e07e85에 보존했다. 당시 보고한 23쪽은 잘못이며 실제 빌드 로그는 24쪽이었다.
- 사용자가 그림 반영을 요청해 `make_paper_figures.py ev_resector`를 추가했다. Figure 18의 (a)는 최대 불균형의 후보 1/8 비율, (b)는 wall time의 후보 1/8 비율로 30조건 모두를 색/마커와 세로 오프셋으로 구별한다. (c)는 overlap off/on의 평균 시간비 6개와 3개 개별 반복 시간비의 최소–최대 whisker다. Whisker는 신뢰구간이 아니다. 점선 1보다 오른쪽이면 기본 구성에 유리하다.
- `analysis/paper_n2.59842747.json`을 입력으로 사용하고 생성 시 96회/48쌍과 provenance를 재검증한다. 새 패널 `paper/figures/ev_resector_{a,b,c,legend}.pdf` 네 개만 추가했다. Figure 17 번호는 유지하며 새 그림은 그 다음이다. 본문에 불균형 정의, 단일 측정/고정 seed, 3회 반복과 whisker 정의를 명시했다. 그림 속 숫자 주석 없이 축 제목만 bold인 기존 스타일을 따른다. 최종 렌더는 `paper/build/resector-page.png`, 독립 미리보기는 `paper/build/resector-preview.pdf`다.
- 최종 검증: 저장 JSON과 현재 96개 로그/48개 비교쌍이 일치한다. 원고 24쪽, prose 8,472단어(Evaluation 3,067), check_style 0건, 미정의 참조/인용과 Overfull 및 TeX 오류 0건이다. paper/main.pdf와 build/main.pdf가 일치한다. 새 그림/본문은 로컬 커밋으로 보존하고 push하지 않는다.

## 후속 변경 — Figure 18 새 실험 구현과 제출 제한 (2026-10-16 08:24 UTC)

- Figure 18까지의 원고/그림은 사용자 요청으로 61d80bc까지 push했다. 사용자가 후보 8개 비교의 시간 이득이 약함을 지적해, 실제 sector 수 적응의 효과를 분리하는 실험을 설계했다(`jobs/figure18_design.json`, 로컬 53876cb). 16 GPU, 적응형 대 고정 16/32/64/128/256개이며 고정 행 비교와 다르다. 행 재계산/후보 8개/overlap/overflow 분할을 공통 유지하고, 최초 적응형 요청이 GPU 수(16) 이상일 때부터 고정 개수를 적용한다. 각 workload의 가장 빠른 고정 개수도 강한 대조군으로 비교한다.
- 사용자가 active goal “새실험 이어서 진행해”로 실행을 요청했다. 구현은 로컬 349a73e다. `src/engine/count_policy.hpp`와 엔진 옵션 `--fixed-sectors`, `--policy-seed`, `--policy-audit`를 추가했다. 기본 policy seed=-1에서는 기존 RNG를 사용한다. 실험 모드는 window 시작/candidate 번호별 독립 stream과 별도 spare-row stream을 사용하며 회로 seed 7은 유지한다. audit는 root의 로컬 row rank/hash/개수 결정을 기록하며 추가 MPI는 없다. 기존 바이너리는 재빌드하지 않고 `bin/pp_engine_n2count`를 수동 빌드했다. 빌드/입력 SHA-256은 `logs/n2count/build_inputs.sha256`, 기존 pp_engine_x/i/iw 해시는 `existing_binaries.sha256`이며 모두 일치한다.
- 실행기 `jobs/n2_count.sh` → `jobs/n2_count.py`는 새 pp_n2count 4노드/4시간 interactive를 한 번 요청하고, 단일 loop/단일 srun만 수행한다. 먼저 warmup 1회, 기존/새 legacy 모드 7회로 14회, 새 matched-stream 입력 검증 7회를 실행한다. 이 gate를 통과해야 예비 36회(KI/TFIM/XXZ × 1e-4/3e-6 × 6정책)를 실행한다. 순서는 seed 180016으로 미리 섞었다. 대규모 본 실험 630회는 자동 시작하지 않고 예비 결과를 먼저 검토한다.
- `analysis/paper_n2count.py`는 완료/실패/미완료/invalid를 구별하고 같은 할당/바이너리/입력인지, 기대값 1e-9 이내와 window schedule, 고정 개수 활성화, bootstrap row hash/항 수가 같은지 검사한다. 전체 6정책 block이 유효해야 시간비를 만든다. C++ control 검증과 Python 분석 거부 테스트 5개 통과, dry-run 예비 36회/6묶음 확인, 기존 바이너리 3개 해시 유지 확인. GPU 호환성 검증은 아직 수행되지 않았다.
- **08:23:10 UTC 실제 salloc 요청이 QOSMaxSubmitJobPerUserLimit로 거부됐다.** interactive MaxSubmitJobsPU=2이고 공유 계정의 non-pp_ interactive 두 개가 사용 중이었다. 새 할당/job은 없고 loop는 종료했다. 다른 사용자의 job에는 손대지 않았다. 기존 대규모 pp_ 8개도 Priority 대기였다. 원고/그림은 이번에 수정하지 않았다.
- 재개: 현재 `squeue`와 `timeout 15 pgrep -u sgkim -af '^python3 jobs/n2_count.py$'`, 필요시 PID의 `/proc/.../cmdline`부터 확인한다. 기존 loop가 없고 quota 여유가 있으면 `bash jobs/n2_count.sh > logs/n2count/loop.log 2>&1`를 unified exec의 장기 session으로 시작한다(yield 1초, 이후 결과만 확인). 이 환경에서 즉시 종료하는 shell의 `nohup setsid ... &`는 최초 시도에 process가 남지 않아 foreground unified exec session을 사용했다. 중복 제출/장기 대기 loop는 금지한다. 성공하면 `logs/n2count/allocation.id`와 `<allocation>/`가 생기고, 읽기는 `PYTHONDONTWRITEBYTECODE=1 python3 analysis/paper_n2count.py --allocation <ID>`다. 실행 중 스크립트/새 바이너리를 바꾸지 않는다.
- 08:27 UTC 후속 goal turn에서도 두 non-pp_ interactive job이 RUNNING이고 새 pp_n2count job/loop는 없었다. 제한이 그대로라 중복 제출하지 않았다. GPU가 필요 없는 실제 엔진 window 선택 통합 검증을 추가했다(`analysis/test_n2_windows.cpp`, 402c1e7). 7개 실제 회로 × 3개 window 시작 × 5개 row 수 = 105조건에서 모든 선택 행과 overflow용 spare 행의 generator 보존, 후보 8개의 표본 최대값이 후보 1개보다 나쁘지 않음, 다른 난수 소비/후보 수/행 수에도 같은 window stream 재현을 검증했다. MPI/CUDA context나 GPU 실행 없이 약 1초 이내 통과했다. 이 결과는 정확성 준비 검사이며 성능 데이터가 아니다. 실행 바이너리와 실험 설계는 그대로다. GPU 실험은 여전히 미시작이다.
