#!/usr/bin/env bash
# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# Create a monthly cost budget with e-mail alerts. Run this ONCE, before the
# first launch.
#
# The expected spend for the ARI GPU conditions is a few dollars. This budget
# is set well above that on purpose: it is there to catch an instance nobody
# terminated, not to police the run.
#
# Usage: ./budget.sh you@example.com [limit-usd]
set -euo pipefail

EMAIL="${1:?usage: budget.sh <email> [limit-usd]}"
LIMIT="${2:-100}"
NAME="semq-ari-gpu"
ACCT=$(aws sts get-caller-identity --query Account --output text)

echo "==> account $ACCT, budget '$NAME' at \$$LIMIT/month, alerts to $EMAIL"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/budget.json" <<EOF
{
  "BudgetName": "$NAME",
  "BudgetLimit": {"Amount": "$LIMIT", "Unit": "USD"},
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
EOF

# Alert on the way up (50%, 80% actual) and once forecast to breach, which is
# the one that gives useful warning while an instance is still running.
cat > "$TMP/notifications.json" <<EOF
[
  {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                    "Threshold": 50, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]},
  {"Notification": {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
                    "Threshold": 80, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]},
  {"Notification": {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN",
                    "Threshold": 100, "ThresholdType": "PERCENTAGE"},
   "Subscribers": [{"SubscriptionType": "EMAIL", "Address": "$EMAIL"}]}
]
EOF

if aws budgets describe-budget --account-id "$ACCT" --budget-name "$NAME" >/dev/null 2>&1; then
  echo "    budget already exists, updating the limit"
  aws budgets update-budget --account-id "$ACCT" --new-budget "file://$TMP/budget.json"
else
  aws budgets create-budget --account-id "$ACCT" \
    --budget "file://$TMP/budget.json" \
    --notifications-with-subscribers "file://$TMP/notifications.json"
  echo "    created"
fi

echo
echo "Budget alerts are lagging indicators -- AWS cost data is hours behind."
echo "The instance-side shutdown timer in bootstrap.sh is the real guard."
