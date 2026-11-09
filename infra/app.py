#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.database_stack import DatabaseStack
from stacks.auth_stack import AuthStack
from stacks.application_stack import ApplicationStack

app = cdk.App()

# CDK context flag: set --context use_cognito=true to enable Cognito ALB auth.
# Default is False -- the app's built-in password login handles auth for MVP.
# Enable Cognito later for subscription model / multi-user accounts.
use_cognito = app.node.try_get_context("use_cognito") in ("true", "True", True)

network = NetworkStack(app, "PropTerminalNetwork")
storage = StorageStack(app, "PropTerminalStorage", vpc=network.vpc, efs_sg=network.efs_sg)

database = DatabaseStack(
    app,
    "PropTerminalDatabase",
    vpc=network.vpc,
    rds_sg=network.rds_sg,
    redis_sg=network.redis_sg,
)

# Cognito auth stack -- only deployed when use_cognito=true
auth_kwargs = {}
if use_cognito:
    auth = AuthStack(app, "PropTerminalAuth")
    auth_kwargs = dict(
        user_pool=auth.user_pool,
        user_pool_client=auth.user_pool_client,
        user_pool_domain=auth.user_pool_domain,
    )

app_stack = ApplicationStack(
    app,
    "PropTerminalApp",
    vpc=network.vpc,
    alb_sg=network.alb_sg,
    fargate_sg=network.fargate_sg,
    efs_file_system=storage.efs_file_system,
    efs_access_point=storage.efs_access_point,
    s3_bucket=storage.s3_bucket,
    secrets=storage.secrets,
    db_secret=database.db_secret,
    redis_endpoint=database.redis_endpoint,
    redis_port=database.redis_port,
    use_cognito=use_cognito,
    **auth_kwargs,
)

app.synth()
