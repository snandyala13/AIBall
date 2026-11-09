# PropTerminal AWS Deployment Checklist

## Prerequisites

- [ ] AWS account created
- [ ] AWS CLI installed and configured (`aws sts get-caller-identity` works)
- [ ] Node.js installed (`node --version` >= 18)
- [ ] Python 3.11+ installed
- [ ] Docker installed and running
- [ ] Domain registered (optional, can add later)

## Step 1: Install CDK

```bash
npm install -g aws-cdk
cd infra && pip install -r requirements.txt
```

Verify: `cdk --version`

## Step 2: Generate secrets

Create your password hash and session secret:

```bash
# Generate PT_PASSWORD_HASH (SHA-256 of your chosen password)
echo -n "your-chosen-password" | shasum -a 256 | cut -d' ' -f1

# Generate SESSION_SECRET (random string)
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Save these values -- you will need them in Step 7.

## Step 3: Bootstrap CDK

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Bootstrap CDK (one-time setup per account/region)
cdk bootstrap aws://$ACCOUNT_ID/us-east-1
```

## Step 4: Synthesize and review

```bash
cd infra
cdk synth --context use_cognito=false
```

This generates CloudFormation templates in `cdk.out/`. Review for sanity.

## Step 5: Deploy infrastructure

```bash
cdk deploy --all --context use_cognito=false --require-approval broadening
```

This creates: VPC, EFS, S3, RDS, Redis, ECR, ECS, ALB, EventBridge rules, CloudWatch.

Note the outputs -- especially `AlbDnsName` and `EcrRepositoryUri`.

## Step 6: Build and push Docker image

```bash
# Get your account ID and ECR URI from CDK outputs
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
ECR_URI=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/propterminal

# Login to ECR
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build the image (from repo root, not deploy/)
cd /path/to/aiBall
docker build -f deploy/Dockerfile -t propterminal .

# Tag and push
docker tag propterminal:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

## Step 7: Fill Secrets Manager values

```bash
# API keys
aws secretsmanager put-secret-value \
  --secret-id propterminal/balldontlie-api-key \
  --secret-string "your-balldontlie-key"

aws secretsmanager put-secret-value \
  --secret-id propterminal/propline-api-key \
  --secret-string "your-propline-key"

aws secretsmanager put-secret-value \
  --secret-id propterminal/odds-api-key \
  --secret-string "your-odds-api-key"

# CTG credentials (JSON)
aws secretsmanager put-secret-value \
  --secret-id propterminal/ctg-credentials \
  --secret-string '{"email":"your@email.com","password":"your-ctg-password"}'

# PropTerminal auth (JSON) -- use values from Step 2
aws secretsmanager put-secret-value \
  --secret-id propterminal/auth \
  --secret-string '{"password_hash":"YOUR_SHA256_HASH","session_secret":"YOUR_RANDOM_SECRET"}'
```

## Step 8: Force ECS deployment

After pushing the image and filling secrets, force ECS to pull the new image:

```bash
aws ecs update-service \
  --cluster propterminal-cluster \
  --service propterminal-web \
  --force-new-deployment
```

Wait for the deployment to stabilize (~2-3 minutes):

```bash
aws ecs wait services-stable \
  --cluster propterminal-cluster \
  --services propterminal-web
```

## Step 9: Verify

```bash
# Get ALB URL from CDK output (or from AWS console)
ALB_URL=$(aws cloudformation describe-stacks \
  --stack-name PropTerminalApp \
  --query "Stacks[0].Outputs[?OutputKey=='AlbDnsName'].OutputValue" \
  --output text)

# Health check
curl http://$ALB_URL/health
# Expected: {"status":"ok"}

# Open in browser
echo "Open: http://$ALB_URL"
# Should see PropTerminal login page
# Login with the password you chose in Step 2
```

## Troubleshooting

### Container won't start

```bash
# Check ECS task logs
aws logs tail /ecs/propterminal --follow

# Check task stopped reason
aws ecs describe-tasks \
  --cluster propterminal-cluster \
  --tasks $(aws ecs list-tasks --cluster propterminal-cluster --service-name propterminal-web --query 'taskArns[0]' --output text)
```

### Health check failing

```bash
# Check if the container is running
aws ecs describe-services \
  --cluster propterminal-cluster \
  --services propterminal-web \
  --query "services[0].{desired:desiredCount,running:runningCount,events:events[:3]}"
```

### Secrets not loading

```bash
# Verify secrets exist and have values
aws secretsmanager get-secret-value --secret-id propterminal/auth --query SecretString --output text
```

## Optional: Add HTTPS (recommended for production)

1. Register a domain in Route 53 (or point your DNS to the ALB)
2. Request an ACM certificate for your domain
3. Add `acm_cert_arn` to CDK context and redeploy
4. Update ALB listener to use HTTPS with the certificate

## Optional: Enable Cognito (Phase 2, subscription model)

```bash
cdk deploy --all --context use_cognito=true
```

This adds the AuthStack and switches ALB to Cognito-authenticated routing.
