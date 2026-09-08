#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Provision one GPU instance for the ARI GPU conditions, with cost guards.
#
# Most runs here cost a few dollars, and the way they become expensive is an
# instance left running. The ARI-D fp32 reference session (run_refs.sh) is the
# exception: it needs a multi-GPU node to hold a 70B in fp32, where the node
# itself is tens of dollars an hour and the run is the expense rather than the
# leak. Read the sizing table below before launching that one.
#
# Either way the guards are the same:
#
#   * instance-initiated-shutdown-behavior=terminate
#   * a shutdown timer armed in user-data, before any work starts
#   * a billing alarm, created separately by budget.sh
#
# Usage:
#   ./launch.sh                 # provision and print the SSH command
#   ./launch.sh --dry-run       # show what would happen, touch nothing
#   INSTANCE_TYPE=g6e.xlarge ./launch.sh
set -euo pipefail

# us-east-1, because that is where the quota is. G/VT on-demand stands at 192
# vCPUs there and g6e.48xlarge needs exactly that; us-east-2 has 4 (a July
# request to raise it to 16 closed without a grant) and us-west-2 has 48.
# The baselines bucket stays in us-east-2 -- cross-region reads cost cents on
# artifacts this size, which is cheaper than waiting on a quota case.
REGION="${AWS_REGION:-us-east-1}"
# Sizing is set by the fp32 reference, not by inference: fp32 is 4 bytes per
# parameter resident, while the checkpoint on disk is the 2-byte bf16 it ships
# as. Both numbers bind -- VRAM decides the instance, disk decides the volume.
#
#   model                    fp32 VRAM   on disk   instance
#   Qwen2.5-3B-Instruct         ~12 GB     ~6 GB   g5.xlarge   (A10G,  24 GB)
#   Qwen2.5-7B-Instruct         ~30 GB    ~15 GB   g6e.xlarge  (L40S,  48 GB)
#   Llama-3.3-70B-Instruct     ~282 GB   ~141 GB   g6e.48xlarge (8xL40S, 384 GB)
#   Qwen2.5-72B-Instruct       ~291 GB   ~145 GB   g6e.48xlarge
#
# The 70B row is why VOLUME_GB is a variable. Two 70B-class checkpoints do not
# fit the 200 GB default, and the HF download fails partway with the node
# already billing. run_refs.sh sets what it needs.
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
VOLUME_GB="${VOLUME_GB:-200}"
# Hard ceiling. The instance terminates itself after this many minutes even if
# everything else fails.
MAX_MINUTES="${MAX_MINUTES:-300}"
# Region-scoped, because a key pair is. The name used to be region-free while
# the private key was written to one path, so launching in a second region
# created a new pair there and overwrote the first region's key on disk --
# losing SSH to anything still running under it.
KEY_NAME="${KEY_NAME:-semq-ari-gpu-${REGION}}"
TAG="${TAG:-semq-ari-gpu}"
# Instance profile carrying AmazonSSMManagedInstanceCore, so the box can be
# driven with `aws ssm send-command` over the AWS API rather than inbound SSH.
# Inbound SSH to this account is unreliable enough that a whole 3-hour run was
# lost to it on 2026-08-08, while the API path did not fail once. Set to "" to
# opt out.
IAM_PROFILE="${IAM_PROFILE:-semq-benchmarks-ec2}"
# Launch into a purchased Capacity Block instead of hunting for on-demand
# capacity. A block is prepaid for a fixed window, so its zone and instance
# type are already decided -- both are read back from the reservation rather
# than trusted to whatever the caller exported, because a mismatch does not
# use the block, it silently falls back to on-demand and fails on capacity.
CAPACITY_RESERVATION="${CAPACITY_RESERVATION:-}"
# gp3 defaults to 125 MB/s, and the reference session writes ~286 GB of
# checkpoint before it can start: 38 minutes of a prepaid 24-hour window
# spent waiting on a volume. 1000 MB/s costs a few dollars for a day.
VOLUME_THROUGHPUT="${VOLUME_THROUGHPUT:-1000}"
VOLUME_IOPS="${VOLUME_IOPS:-16000}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
run() { if (( DRY_RUN )); then echo "  [dry-run] $*"; else eval "$@"; fi }

say "region=$REGION type=$INSTANCE_TYPE volume=${VOLUME_GB}GB kill-after=${MAX_MINUTES}min"

if ! aws sts get-caller-identity --region "$REGION" >/dev/null 2>&1; then
  echo "ERROR: no working AWS credentials. Run 'aws configure' or" >&2
  echo "'aws configure sso && aws sso login', then retry." >&2
  exit 1
fi
say "authenticated as $(aws sts get-caller-identity --query Arn --output text)"

# --- AMI -------------------------------------------------------------------
# Deep Learning AMIs are published as SSM public parameters. The exact names
# change as versions roll, so try the current ones and fall back to a search.
say "resolving Deep Learning AMI"
AMI=""
for P in \
  "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id" \
  "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id"
do
  AMI=$(aws ssm get-parameter --name "$P" --region "$REGION" \
        --query 'Parameter.Value' --output text 2>/dev/null || true)
  [[ -n "$AMI" && "$AMI" != "None" ]] && { say "  $P -> $AMI"; break; }
done
if [[ -z "$AMI" || "$AMI" == "None" ]]; then
  say "  SSM lookup failed, searching public AMIs"
  AMI=$(aws ec2 describe-images --region "$REGION" --owners amazon \
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
              "Name=state,Values=available" \
    --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text)
  say "  found $AMI"
fi
[[ -z "$AMI" || "$AMI" == "None" ]] && { echo "ERROR: no AMI found" >&2; exit 1; }

# --- key pair --------------------------------------------------------------
KEY_PATH="$HOME/.ssh/${KEY_NAME}.pem"
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
  say "key pair $KEY_NAME exists"
  [[ -f "$KEY_PATH" ]] || echo "  WARNING: $KEY_PATH missing locally; you cannot SSH in." >&2
elif [[ -f "$KEY_PATH" ]]; then
  # The pair is absent from this region but the path is taken: writing here
  # would destroy a private key that is still the only copy for some other
  # region. Never overwrite; make the operator choose.
  echo "ERROR: $KEY_PATH exists, but no key pair named $KEY_NAME is in" >&2
  echo "$REGION. Writing a new one would overwrite a key you cannot" >&2
  echo "recover. Set KEY_NAME to something else, or delete that file if" >&2
  echo "you are sure no instance still needs it." >&2
  exit 1
else
  say "creating key pair $KEY_NAME -> $KEY_PATH"
  run "aws ec2 create-key-pair --key-name '$KEY_NAME' --region '$REGION' \
        --query KeyMaterial --output text > '$KEY_PATH' && chmod 400 '$KEY_PATH'"
fi

# --- security group --------------------------------------------------------
VPC=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
      --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --region "$REGION" \
     --filters "Name=group-name,Values=$TAG" "Name=vpc-id,Values=$VPC" \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
if [[ "$SG" == "None" || -z "$SG" ]]; then
  say "creating security group $TAG in $VPC"
  MYIP=$(curl -s https://checkip.amazonaws.com | tr -d '\n')
  run "SG=\$(aws ec2 create-security-group --group-name '$TAG' \
        --description 'SEMQ ARI GPU runs' --vpc-id '$VPC' --region '$REGION' \
        --query GroupId --output text) && \
       aws ec2 authorize-security-group-ingress --group-id \$SG --protocol tcp \
        --port 22 --cidr '${MYIP}/32' --region '$REGION' && echo \$SG"
  (( DRY_RUN )) || SG=$(aws ec2 describe-security-groups --region "$REGION" \
      --filters "Name=group-name,Values=$TAG" "Name=vpc-id,Values=$VPC" \
      --query 'SecurityGroups[0].GroupId' --output text)
  say "  SSH restricted to ${MYIP:-your IP}/32"
else
  say "security group $TAG exists ($SG)"
fi

# --- launch ----------------------------------------------------------------
USER_DATA=$(sed "s/__MAX_MINUTES__/$MAX_MINUTES/" "$(dirname "$0")/bootstrap.sh" | base64)

if [[ -n "$IAM_PROFILE" ]]; then
  PROFILE_ARG=(--iam-instance-profile "Name=$IAM_PROFILE")
  say "attaching instance profile $IAM_PROFILE (SSM)"
else
  # An empty array expands to an unbound-variable error under `set -u` on
  # bash 3.2 (macOS). Hold a no-op argument instead of nothing.
  PROFILE_ARG=(--region "$REGION")
fi

# Try every availability zone that offers the type, rather than letting EC2
# pick one and taking its answer as final. The large GPU types are capacity
# constrained: g6e.48xlarge returns InsufficientInstanceCapacity from most
# zones on most days, and the zone that has it changes. Without the loop a
# launch fails on a zone that was never asked about the other three.
CR_ARG=()
if [[ -n "$CAPACITY_RESERVATION" ]]; then
  read -r CR_AZ CR_TYPE CR_STATE CR_AVAIL < <(aws ec2 describe-capacity-reservations \
    --region "$REGION" --capacity-reservation-ids "$CAPACITY_RESERVATION" \
    --query 'CapacityReservations[0].[AvailabilityZone,InstanceType,State,AvailableInstanceCount]' \
    --output text) || { echo "ERROR: cannot read $CAPACITY_RESERVATION" >&2; exit 1; }
  say "capacity block $CAPACITY_RESERVATION: $CR_TYPE in $CR_AZ, state=$CR_STATE, free=$CR_AVAIL"
  [[ "$CR_STATE" == "active" ]] || { echo "ERROR: reservation is '$CR_STATE', not 'active'. A block
is only launchable inside its window; 'scheduled' means it has not opened yet." >&2; exit 1; }
  [[ "$CR_AVAIL" -gt 0 ]] || { echo "ERROR: no free capacity left in the block." >&2; exit 1; }
  if [[ "$CR_TYPE" != "$INSTANCE_TYPE" ]]; then
    say "  overriding INSTANCE_TYPE=$INSTANCE_TYPE with the block's $CR_TYPE"
    INSTANCE_TYPE="$CR_TYPE"
  fi
  # Targeting the reservation is necessary but not sufficient: RunInstances
  # rejects a capacity-block reservation as "market type (purchasing) option
  # is not valid" unless the request also declares the capacity-block market.
  # Prepaid block capacity is its own market, not on-demand aimed at a
  # reservation.
  CR_ARG=(--capacity-reservation-specification \
          "CapacityReservationTarget={CapacityReservationId=$CAPACITY_RESERVATION}" \
          --instance-market-options "MarketType=capacity-block")
  AZS=("$CR_AZ")           # the block lives in one zone; do not sweep
else
  CR_ARG=(--region "$REGION")   # no-op filler; empty arrays break bash 3.2 -u
  AZS=()
while IFS= read -r az; do
  [[ -n "$az" ]] && AZS+=("$az")
done < <(aws ec2 describe-instance-type-offerings --region "$REGION" \
  --location-type availability-zone \
  --filters "Name=instance-type,Values=$INSTANCE_TYPE" \
  --query 'InstanceTypeOfferings[].Location' --output text | tr '\t' '\n' | sort)
fi
(( ${#AZS[@]} )) || { echo "ERROR: $INSTANCE_TYPE is offered in no AZ of $REGION" >&2; exit 1; }
say "zones offering $INSTANCE_TYPE: ${AZS[*]}"

say "launching $INSTANCE_TYPE"
if (( DRY_RUN )); then
  echo "  [dry-run] ami=$AMI sg=$SG key=$KEY_NAME"
  echo "  [dry-run] type=$INSTANCE_TYPE zones=${AZS[*]}"
  echo "  [dry-run] volume=${VOLUME_GB}GB gp3 ${VOLUME_THROUGHPUT}MB/s ${VOLUME_IOPS}iops"
  [[ -n "$CAPACITY_RESERVATION" ]] \
    && echo "  [dry-run] into block $CAPACITY_RESERVATION ($CR_STATE, $CR_AVAIL free)" \
    || echo "  [dry-run] on-demand, sweeping ${#AZS[@]} zone(s)"
  exit 0
fi

IID=""
for AZ in "${AZS[@]}"; do
  SUBNET=$(aws ec2 describe-subnets --region "$REGION" \
    --filters "Name=availability-zone,Values=$AZ" "Name=default-for-az,Values=true" \
    --query 'Subnets[0].SubnetId' --output text 2>/dev/null)
  [[ -z "$SUBNET" || "$SUBNET" == "None" ]] && { say "  $AZ: no default subnet, skipping"; continue; }
  say "  trying $AZ ($SUBNET)"
  IID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
    "${PROFILE_ARG[@]}" "${CR_ARG[@]}" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_GB,\"VolumeType\":\"gp3\",\"Iops\":$VOLUME_IOPS,\"Throughput\":$VOLUME_THROUGHPUT,\"DeleteOnTermination\":true}}]" \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$TAG},{Key=Project,Value=semq-ari}]" \
    --query 'Instances[0].InstanceId' --output text 2>/tmp/semq-launch-err) && break
  IID=""
  if grep -q InsufficientInstanceCapacity /tmp/semq-launch-err; then
    if [[ -n "$CAPACITY_RESERVATION" ]]; then
      echo "ERROR: the capacity block reported no capacity. This should not" >&2
      echo "happen inside an active window -- check the reservation before" >&2
      echo "retrying, rather than falling back to on-demand." >&2
      exit 1
    fi
    say "    no capacity in $AZ"
  else
    cat /tmp/semq-launch-err >&2; exit 1     # a real error, not scarcity
  fi
done

if [[ -z "$IID" ]]; then
  echo "ERROR: no capacity for $INSTANCE_TYPE in any zone of $REGION." >&2
  echo "Retry later, or try a different region -- but check the quota there" >&2
  echo "first: this type needs 192 vCPUs of G/VT on-demand." >&2
  exit 1
fi

say "instance $IID launching, waiting for it to be running"
aws ec2 wait instance-running --instance-ids "$IID" --region "$REGION"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --region "$REGION" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

cat <<EOF

$(say "ready")

  instance   $IID
  address    $IP
  self-terminates in ${MAX_MINUTES} minutes, no matter what

  ssh -i $KEY_PATH ubuntu@$IP

  Setup runs in the background on boot. Watch it finish with:
    ssh -i $KEY_PATH ubuntu@$IP 'tail -f /var/log/semq-bootstrap.log'

  Then run the experiments:
    ssh -i $KEY_PATH ubuntu@$IP 'bash ~/semq-ari-benchmark/infra/run_gpu.sh'

  Pull the results back:
    ./fetch_results.sh $IP

  TERMINATE WHEN DONE -- do not rely on the timer:
    aws ec2 terminate-instances --instance-ids $IID --region $REGION

EOF
echo "$IID" > "$(dirname "$0")/.last-instance"
