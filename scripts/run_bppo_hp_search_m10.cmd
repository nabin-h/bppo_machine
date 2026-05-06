@echo off
setlocal

pushd "%~dp0\.."

set "PY=.venv311\Scripts\python.exe"
set "DATASET=datasets_m10s10_poscost_v2_mix567_r20\MM10\mix_t5_t6_t7_equal_rnd20\traj_mixthr_MM10_mix_t5_t6_t7_equal_rnd20_20000_minR2.pkl"
set "RUNROOT=M10 runs\bppo_hp"
set "COMMON=--seed 123 --env_name m10s10_poscost_v2 --num_machines 10 --horizon 100 --episodes 100 --discount 0.95 --device cuda --v_steps 50000 --v_hidden_dim 256 --v_depth 2 --v_lr 0.0001 --v_batch_size 512 --q_bc_steps 50000 --q_hidden_dim 256 --q_depth 2 --q_lr 0.0001 --q_batch_size 512 --target_update_freq 2 --tau 0.005 --bc_hidden_dim 256 --bc_depth 2 --bc_lr 0.0001 --bc_batch_size 512 --bppo_hidden_dim 256 --bppo_depth 2 --bppo_lr 0.0001 --bppo_batch_size 512 --clip_ratio 0.35 --decay 0.96 --omega 0.9 --eval_interval 250 --checkpoint_interval 500 --log_interval 100 --policy_path policies\M10S10_poscost_v2_threshold6_policy.pkl --extra_policy_paths policies\M10S10_poscost_v2_threshold5_policy.pkl policies\M10S10_poscost_v2_threshold7_policy.pkl"

call :runone bc20_bppo5_ent0 20000 5000 0.0
call :runone bc20_bppo5_ent1 20000 5000 0.01
call :runone bc20_bppo10_ent0 20000 10000 0.0
call :runone bc20_bppo10_ent1 20000 10000 0.01
call :runone bc35_bppo5_ent0 35000 5000 0.0
call :runone bc35_bppo5_ent1 35000 5000 0.01
call :runone bc35_bppo10_ent0 35000 10000 0.0
call :runone bc35_bppo10_ent1 35000 10000 0.01
call :runone bc50_bppo5_ent0 50000 5000 0.0
call :runone bc50_bppo5_ent1 50000 5000 0.01
call :runone bc50_bppo10_ent0 50000 10000 0.0
call :runone bc50_bppo10_ent1 50000 10000 0.01

popd
exit /b 0

:runone
echo ==================================================
echo Running %1
echo ==================================================
"%PY%" train_bppo_machine.py --dataset_path "%DATASET%" --save_dir "%RUNROOT%\%1\MM10\mix_t567_e_r20" %COMMON% --bc_steps %2 --bppo_steps %3 --entropy_weight %4
if errorlevel 1 exit /b 1
exit /b 0
