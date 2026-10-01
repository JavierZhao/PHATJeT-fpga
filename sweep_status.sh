#!/bin/sh
# Status of the PHAT-JeT QAT grid/tokenizer sweep on Nautilus (cms-ml).
#
# NOTE on job status: the sweep jobs pipe python into `tee`, so a job that
# crashed mid-run can still report Succeeded unless the spec sets
# `set -o pipefail`. Trust the epoch counter below, not the k8s STATUS column.
#
# Usage: sh sweep_status.sh
set -u
NS=cms-ml

printf '%-30s %-10s %-12s %-8s %s\n' JOB STATUS LAST_EPOCH BEST_ACC BEST_CKPT
for j in $(kubectl get jobs -n "$NS" -o name 2>/dev/null | grep 'zz-phatjet' | cut -d/ -f2); do
    st=$(kubectl get job "$j" -n "$NS" -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
    [ -z "$st" ] && st=Running
    log=$(kubectl logs -n "$NS" "job/$j" 2>/dev/null)
    ep=$(printf '%s' "$log" | grep -oE '^Epoch [0-9]+/[0-9]+' | tail -1 | awk '{print $2}')
    [ -z "$ep" ] && ep='-'
    acc=$(printf '%s' "$log" | grep -oE 'val_accuracy: [0-9.]+' | tail -1 | awk '{print $2}')
    [ -z "$acc" ] && acc='-'
    ck=$(printf '%s' "$log" | grep -oE 'ebops: [0-9.]+' | tail -1 | awk '{print $2}')
    [ -z "$ck" ] && ck='-'
    printf '%-30s %-10s %-12s %-8s ebops=%s\n' "$j" "$st" "$ep" "$acc" "$ck"
done

# Last-epoch accuracy is noisy: beta is ramping, so accuracy and EBOPs both move
# by design. The meaningful comparison between arms is their Pareto fronts --
# accuracy at matched EBOPs. Those live on the PVC as checkpoint filenames.
echo
echo "=== Pareto fronts (accuracy at matched EBOPs is the real comparison) ==="
pod=$(kubectl get pods -n "$NS" -l 'zz-sweep in (phatjet-grid,phatjet-phase2,phatjet-phase3)' \
        --field-selector=status.phase=Running -o name 2>/dev/null | head -1)
[ -z "$pod" ] && pod=$(kubectl get pods -n "$NS" -l job-name=zz-phatjet-qat-150-kt-max \
        --field-selector=status.phase=Running -o name 2>/dev/null | head -1)
if [ -z "$pod" ]; then
    echo "(no running pod to read the PVC through)"
else
    kubectl exec -n "$NS" "$pod" -- sh -c '
      for d in /data/phatjet-fpga/runs/*/pareto; do
        [ -d "$d" ] || continue
        run=$(basename $(dirname "$d"))
        n=$(ls -1 "$d" 2>/dev/null | wc -l)
        best=$(ls -1 "$d" 2>/dev/null | sed -n "s/.*val_acc=\([0-9.]*\)-ebops=\([0-9]*\).*/\1 \2/p" \
               | sort -rn | head -1)
        printf "%-28s %3s ckpts   best(acc ebops)= %s\n" "$run" "$n" "${best:--}"
      done' 2>/dev/null || echo "(exec failed)"
fi
