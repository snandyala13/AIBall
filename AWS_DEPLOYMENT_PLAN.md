# PropTerminal AWS Production Deployment -- Real-Time Betting Intelligence Platform

## Context

PropTerminal is a real-time NBA betting intelligence system that combines ML-based stat prediction with cross-book line shopping across 19+ sportsbooks. The system runs a LightGBM v3 classifier model once per game, caches the prediction, then recalculates edge every 30 seconds as odds change across books. During live games, it ingests box scores and updates projections based on actual game state.

**What makes it different:** Most services give you pre-game picks. PropTerminal continuously monitors odds across all books, recalculates edge in real-time, and during games tracks pace, injuries, foul trouble, and blowout risk to update predictions live.

**Architecture:** FastAPI server, per-prop LightGBM classifier (v3_classifier_holdout), PropLine for 19+ book odds, RDS PostgreSQL for shared state, ElastiCache Redis for prediction cache, built-in password login (Phase 1) with optional Cognito for future subscriptions (Phase 2).

**Model:** v3_classifier_holdout -- LightGBM binary classifiers per prop type with isotonic/logistic calibration. Empirical pick filter (`outputs/empirical_filter.json`) for post-prediction filtering.

---

## Authentication: Two-Phase Approach

### Phase 1: Password Login (current, MVP)
- Built-in login page at `/login`
- Session-based auth via `itsdangerous` signed cookies
- Single shared password set via `PT_PASSWORD_HASH` env var (SHA-256)
- `SESSION_SECRET` env var for cookie signing
- No Cognito required -- deploy with `--context use_cognito=false`

### Phase 2: Cognito User Accounts (future, subscriptions)
- Deploy AuthStack with `--context use_cognito=true`
- ALB-level Cognito auth with per-user accounts
- Enable when ready for multi-user subscription model
- No code changes needed -- just flip the CDK context flag

---

## Core Real-Time Architecture

```
+--------------------------------------------------------+
|                    GAME DAY FLOW                        |
|                                                         |
|  PRE-GAME (12pm-7pm ET)                                |
|  +------------------+    +----------------------+       |
|  | Analyze Games    |--->| Prediction Cache     |       |
|  | (LightGBM v3 1x) |    | (Redis)              |       |
|  | ~15s per game    |    | classifier probs     |       |
|  +------------------+    +----------+-----------+       |
|                                     |                    |
|  CONTINUOUS (every 30s while server is up)              |
|  +------------------+    +----------v-----------+       |
|  | PropLine Bulk   |--->| Edge Recalculator    |       |
|  | (ALL books, 1s) |    | cached adj + fresh    |       |
|  +------------------+    | odds -> new edge/kelly|       |
|                          | + EV per book         |       |
|                          | <5ms total            |       |
|                          +----------+-----------+       |
|                                     |                    |
|  DURING GAMES (7pm-2am ET)         |                    |
|  +------------------+    +----------v-----------+       |
|  | Live Box Scores |--->| In-Game Projections  |       |
|  | (BDL, 60s)     |    | pace-adjusted, blowout|       |
|  +------------------+    | foul trouble, injuries|       |
|                          +----------+-----------+       |
|                                     |                    |
|                          +----------v-----------+       |
|                          | Dashboard (live)      |       |
|                          | password-protected    |       |
|                          | auto-updates every 30s|       |
|                          +----------------------+       |
+--------------------------------------------------------+
```

---

## Stack Structure (4-5 CDK Stacks)

### Stack 1: `NetworkStack`

```
VPC (2 AZs)
+-- Public subnets (ALB)
+-- Private subnets (Fargate, RDS, Redis)
+-- NAT Gateway (1 AZ -- saves $32/mo vs 2)
+-- Security Groups:
|   +-- ALB SG: inbound 80/443 from 0.0.0.0/0
|   +-- Fargate SG: inbound 8000 from ALB SG, all outbound
|   +-- EFS SG: inbound 2049 (NFS) from Fargate SG
|   +-- RDS SG: inbound 5432 (PostgreSQL) from Fargate SG
|   +-- Redis SG: inbound 6379 from Fargate SG
+-- No VPC endpoints (NAT handles all outbound)
```

Fargate tasks in **private subnets** -- internet via NAT Gateway. RDS and Redis also in private subnets, only accessible from Fargate.

### Stack 2: `StorageStack`

```
EFS File System (Bursting throughput, IA lifecycle 30 days)
+-- /data/outputs/                         -- Prediction logs, empirical filter
+-- /data/models/v3_classifier_holdout/    -- Per-prop classifiers (primary)
+-- /data/models/v2_final/                 -- Fallback models

S3 Bucket: aiball-artifacts-{account_id}
+-- models/                     -- Model versioning (S3 versioning enabled)
+-- prediction-logs/            -- Nightly JSONL archival
+-- grading-reports/            -- Historical grading + CLV data
+-- training-data/              -- Backtest JSON archives

Secrets Manager:
+-- propterminal/balldontlie-api-key
+-- propterminal/propline-api-key
+-- propterminal/odds-api-key
+-- propterminal/ctg-credentials       -- {email, password} JSON
+-- propterminal/auth                  -- {password_hash, session_secret} JSON
```

### Stack 3: `DatabaseStack`

```
RDS PostgreSQL 15
+-- Instance: db.t4g.micro (ARM, 2 vCPU, 1GB RAM)
+-- Storage: 20GB gp3
+-- Multi-AZ: false (cost optimization)
+-- Database: propterminal
+-- Credentials: auto-generated -> Secrets Manager
+-- Security group: rds_sg
+-- Backup retention: 7 days
+-- Deletion protection: false (dev)
+-- Replaces: SQLite cache.db + lines.db

ElastiCache Redis 7.x
+-- Node type: cache.t4g.micro (ARM)
+-- Nodes: 1
+-- Security group: redis_sg
+-- Subnet group: private subnets
+-- Replaces: in-memory _prediction_cache, _analysis_cache, _previous_lines
```

### Stack 4: `AuthStack` (OPTIONAL -- Phase 2 only)

Only deployed when `--context use_cognito=true`. Not needed for MVP.

```
Cognito User Pool: propterminal-users
+-- Self sign-up: disabled (admin creates users)
+-- Sign-in: email
+-- MFA: optional (TOTP)
+-- User Pool Client + Domain for ALB integration
```

### Stack 5: `ApplicationStack`

```
ECR Repository: propterminal

ECS Cluster: propterminal-cluster
+-- Service: propterminal-web (Fargate)
|   +-- Task Definition:
|   |   +-- CPU: 1024 (1 vCPU), Memory: 2048 MB
|   |   +-- Platform: LINUX/ARM64 (Graviton -- 20% cheaper)
|   |   +-- Container:
|   |   |   +-- Port: 8000
|   |   |   +-- Health check: curl -f http://localhost:8000/health
|   |   |   +-- Environment:
|   |   |   |   +-- APP_MODE=web
|   |   |   |   +-- EFS_MOUNT=/data
|   |   |   |   +-- AIBALL_MODELS_DIR=/data/models/v3_classifier_holdout
|   |   |   |   +-- REDIS_URL=redis://{endpoint}:6379/0
|   |   |   |   +-- S3_BUCKET={bucket_name}
|   |   |   +-- Secrets (Secrets Manager):
|   |   |   |   +-- BALLDONTLIE_API_KEY, PROPLINE_API_KEY, ODDS_API_KEY
|   |   |   |   +-- CTG_EMAIL, CTG_PASSWORD
|   |   |   |   +-- PT_PASSWORD_HASH, SESSION_SECRET
|   |   |   |   +-- DB_HOST, DB_PORT, DB_USERNAME, DB_PASSWORD
|   |   |   +-- Mounts: EFS -> /data
|   |   +-- Execution Role: ECR pull, Secrets read, CloudWatch write
|   +-- Desired Count: 1 (scale to 2+ for multi-user)
|   +-- Private subnets (internet via NAT)
|   +-- Scheduled scaling: up at 11am ET, down at 3am ET
|
+-- Task Definition: propterminal-batch (scheduled jobs)
|   +-- CPU: 512, Memory: 1024
|   +-- Same image, different CMD
|   +-- Used by EventBridge
|
+-- Scheduled Tasks (EventBridge -> ECS RunTask):
    +-- pre-game-analyze:  4pm ET daily   -- CMD: analyze-all
    +-- grade-predictions: 2am ET daily   -- CMD: grade
    +-- retrain-check:     10am ET daily  -- CMD: check-drift
    +-- retrain-monthly:   1st of month   -- CMD: retrain

ALB (Application Load Balancer):
+-- Phase 1 (use_cognito=false): HTTP:80 -> forward to target group
|   App handles auth via login page + session cookies
+-- Phase 2 (use_cognito=true): HTTPS:443 -> Cognito auth -> forward
|   /health bypasses auth for ALB health checks
+-- Health Check: /health, 200 OK

CloudWatch:
+-- Log Groups: /ecs/propterminal, /ecs/propterminal-batch (30 day retention)
+-- Dashboard: propterminal-ops
|   +-- ECS CPU/Memory
|   +-- ALB request count + latency (P50, P99)
|   +-- 5xx error count
|   +-- PropLine API call count
|   +-- Predictions per day
+-- Alarms:
|   +-- WebUnhealthy: ALB unhealthy hosts > 0 for 5min
|   +-- High5xx: 5xx count > 5 in 5min
|   +-- GradingMissed: no grading metric in 26hrs
|   +-- HitRateDrift: 7-day hit rate < 48%
|   +-- RetrainFailed: batch exit code != 0
+-- SNS Topic: propterminal-alerts -> email
```

---

## Cost Estimate (Monthly)

| Service | Config | Cost |
|---------|--------|------|
| **ECS Fargate** | 1 vCPU, 2GB, ~14hr/day x 30 days | ~$22 |
| **ALB** | 1 ALB + minimal LCU | ~$18 |
| **NAT Gateway** | 1 AZ, ~14hr/day | ~$32 |
| **RDS PostgreSQL** | db.t4g.micro, 20GB | ~$13 |
| **ElastiCache Redis** | cache.t4g.micro | ~$12 |
| **EFS** | 2GB standard + IA | ~$1 |
| **ECR** | <1GB image | ~$0.10 |
| **S3** | 3GB with versioning | ~$0.10 |
| **Secrets Manager** | 6 secrets | ~$3.00 |
| **CloudWatch** | Logs + 5 alarms + dashboard | ~$6 |
| **Route 53** | 1 hosted zone (optional) | ~$0.50 |
| **Total (Phase 1, no Cognito)** | | **~$107/mo** |

**Cost levers:**
- Off-season (Jun-Sep): Scale Fargate + NAT to 0 -> ~$25/mo (just RDS/Redis standby)
- Fargate Spot for batch jobs: 70% discount
- ARM64/Graviton: already included everywhere (20% cheaper)
- RDS stop/start: Can stop RDS during off-season ($0 compute, pay storage only)

---

## Scaling

| Users | ECS Tasks | Redis | RDS | Monthly Cost |
|-------|-----------|-------|-----|-------------|
| 1 | 1 | cache.t4g.micro | db.t4g.micro | ~$107 |
| 5-10 | 2 | cache.t4g.micro | db.t4g.micro | ~$130 |
| 50+ | 3-5 | cache.t4g.small | db.t4g.small | ~$200 |
| 500+ | Auto-scale | cache.m7g.medium | db.t4g.medium | ~$400 |

All scaling is config changes -- no re-architecture needed.

---

## Model Retraining Strategy

Training runs in <10 seconds on Fargate batch (1 vCPU). No SageMaker needed.

| Trigger | Detection | Action |
|---------|-----------|--------|
| **Monthly calendar** | EventBridge cron (1st of month) | Auto-retrain, sync to S3 |
| **Performance drift** | 7-day hit rate < 48% for 3 consecutive days | CloudWatch alarm -> alert |
| **Data events** | All-Star break, trade deadline, playoffs | Manual retrain |
| **Season start** | October (cold start) | Pre-load previous season + retrain |

---

## CI/CD: GitHub Actions -> ECR -> ECS

```yaml
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
      - uses: aws-actions/amazon-ecr-login@v2
      - run: |
          docker build -f deploy/Dockerfile -t $ECR_REGISTRY/propterminal:${{ github.sha }} .
          docker push $ECR_REGISTRY/propterminal:${{ github.sha }}
          docker tag ... :latest && docker push ... :latest
          aws ecs update-service --cluster propterminal-cluster --service propterminal-web --force-new-deployment
```

Zero-downtime deploys via ALB health check draining.

---

## CDK Project Structure

```
infra/
+-- app.py                       # CDK entry point (4-5 stacks, Cognito optional)
+-- cdk.json
+-- requirements.txt
+-- stacks/
    +-- network_stack.py         # VPC, subnets, NAT, security groups
    +-- storage_stack.py         # EFS, S3, Secrets Manager
    +-- database_stack.py        # RDS PostgreSQL, ElastiCache Redis
    +-- auth_stack.py            # Cognito (Phase 2, optional)
    +-- application_stack.py     # ECR, ECS, ALB, EventBridge, CloudWatch

deploy/
+-- Dockerfile
+-- entrypoint.sh
+-- DEPLOY_CHECKLIST.md          # Step-by-step deployment guide

.github/workflows/deploy.yml    # CI/CD pipeline
.dockerignore
```

---

## Implementation Order

1. **Dockerfile + `/health` endpoint** -- containerize, verify locally
2. **`NetworkStack`** -- VPC, subnets, NAT, security groups
3. **`StorageStack`** -- EFS, S3, Secrets Manager (includes propterminal/auth)
4. **`DatabaseStack`** -- RDS PostgreSQL, ElastiCache Redis
5. **`ApplicationStack`** -- ECR, ECS, ALB (no Cognito), EventBridge, CloudWatch
6. **`cdk deploy --all --context use_cognito=false`** -- deploy everything
7. **Fill secrets** -- API keys + PT_PASSWORD_HASH + SESSION_SECRET
8. **Build and push Docker image** -- ECR push, ECS force deploy
9. **Verify** -- hit ALB URL, see login page, authenticate, analyze games

## Verification

1. `docker build -f deploy/Dockerfile -t propterminal .` -- local build
2. `cd infra && cdk synth --context use_cognito=false` -- verify CloudFormation templates
3. `cdk deploy --all --context use_cognito=false` -- deploy to AWS
4. Hit ALB URL -> see PropTerminal login page
5. Login with password -> dashboard loads, polling starts at 30s
6. `/api/live-predictions` returns real-time edge data
7. Wait for grading cron -> check CloudWatch logs + S3
8. Scale ECS to 2 tasks -> verify both serve requests (shared Redis/RDS)

---

## Key Files

| File | Purpose |
|------|---------|
| `web/app.py` | Routes, login page, session auth middleware |
| `web/engine.py` | Prediction cache, `recalculate_edges_for_game()`, `get_live_predictions()`, CLV logging |
| `web/static/app.js` | 30s polling, `updatePicksWithFreshEdge()`, line shopping, live game overlay |
| `v2/predictor.py` | `AIBALL_MODELS_DIR` env var, per-prop v3 classifier loading |
| `v2/engine.py` | Team/travel/derived feature builders wired into inference |
| `v2/live_game.py` | Live box score tracking, pace projection, event detection |
| `v2/team_features.py` | Team four factors, opponent defense, matchup features |
| `v2/travel_features.py` | Travel distance, timezone, altitude, calendar features |
| `v2/derived_features.py` | FTA rates, home/away splits, usage trends, opponent history |
| `v2/ref_features.py` | Referee tendency features |
| `v2/pick_filter.py` | Empirical pick filter using `outputs/empirical_filter.json` |
| `grade_predictions.py` | Nightly grading + CLV calculation |
| `cache_db.py` | SQLite cache (local dev) / PostgreSQL adapter (prod) |
