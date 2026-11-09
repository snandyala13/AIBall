from typing import Optional

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    aws_applicationautoscaling as appscaling,
    aws_cloudwatch as cw,
    aws_cognito as cognito,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_efs as efs,
    aws_elasticloadbalancingv2 as elbv2,
    aws_elasticloadbalancingv2_actions as elbv2_actions,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_logs as logs,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    aws_cloudwatch_actions as cw_actions,
)
from constructs import Construct


class ApplicationStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        alb_sg: ec2.ISecurityGroup,
        fargate_sg: ec2.ISecurityGroup,
        efs_file_system: efs.IFileSystem,
        efs_access_point: efs.IAccessPoint,
        s3_bucket: s3.IBucket,
        secrets: dict[str, secretsmanager.ISecret],
        db_secret: secretsmanager.ISecret,
        redis_endpoint: str,
        redis_port: str,
        user_pool: Optional[cognito.IUserPool] = None,
        user_pool_client: Optional[cognito.IUserPoolClient] = None,
        user_pool_domain: Optional[cognito.IUserPoolDomain] = None,
        use_cognito: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        alert_email = self.node.try_get_context("alert_email") or ""

        # ------------------------------------------------------------------
        # ECR Repository
        # ------------------------------------------------------------------
        # Import existing ECR repo (created by previous deploy, retained on delete)
        ecr_repo = ecr.Repository.from_repository_name(
            self, "PropTerminalRepo", "propterminal"
        )

        # ------------------------------------------------------------------
        # ECS Cluster
        # ------------------------------------------------------------------
        cluster = ecs.Cluster(
            self,
            "PropTerminalCluster",
            cluster_name="propterminal-cluster",
            vpc=vpc,
            container_insights=True,
        )

        # ------------------------------------------------------------------
        # Shared EFS volume configuration
        # ------------------------------------------------------------------
        efs_volume_config = ecs.Volume(
            name="efs-data",
            efs_volume_configuration=ecs.EfsVolumeConfiguration(
                file_system_id=efs_file_system.file_system_id,
                transit_encryption="ENABLED",
                authorization_config=ecs.AuthorizationConfig(
                    access_point_id=efs_access_point.access_point_id,
                    iam="ENABLED",
                ),
            ),
        )

        # ------------------------------------------------------------------
        # Shared secrets mapping (API keys + database credentials + auth)
        # ------------------------------------------------------------------
        secret_env = {
            "BALLDONTLIE_API_KEY": ecs.Secret.from_secrets_manager(
                secrets["balldontlie"]
            ),
            "PROPLINE_API_KEY": ecs.Secret.from_secrets_manager(
                secrets["propline"]
            ),
            "ODDS_API_KEY": ecs.Secret.from_secrets_manager(
                secrets["odds_api"]
            ),
            "CTG_EMAIL": ecs.Secret.from_secrets_manager(
                secrets["ctg"], field="email"
            ),
            "CTG_PASSWORD": ecs.Secret.from_secrets_manager(
                secrets["ctg"], field="password"
            ),
            "PT_PASSWORD_HASH": ecs.Secret.from_secrets_manager(
                secrets["pt_auth"], field="password_hash"
            ),
            "SESSION_SECRET": ecs.Secret.from_secrets_manager(
                secrets["pt_auth"], field="session_secret"
            ),
            "DB_HOST": ecs.Secret.from_secrets_manager(db_secret, field="host"),
            "DB_PORT": ecs.Secret.from_secrets_manager(db_secret, field="port"),
            "DB_USERNAME": ecs.Secret.from_secrets_manager(db_secret, field="username"),
            "DB_PASSWORD": ecs.Secret.from_secrets_manager(db_secret, field="password"),
        }

        # Shared plain-text environment variables
        shared_env = {
            "REDIS_URL": f"redis://{redis_endpoint}:{redis_port}/0",
            "AIBALL_MODELS_DIR": "/data/models/v3_classifier_holdout",
        }

        # ------------------------------------------------------------------
        # CloudWatch Log Groups
        # ------------------------------------------------------------------
        web_log_group = logs.LogGroup(
            self,
            "WebLogGroup",
            log_group_name="/ecs/propterminal",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        batch_log_group = logs.LogGroup(
            self,
            "BatchLogGroup",
            log_group_name="/ecs/propterminal-batch",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ------------------------------------------------------------------
        # Task Definition: propterminal-web
        # ------------------------------------------------------------------
        web_task_def = ecs.FargateTaskDefinition(
            self,
            "WebTaskDef",
            family="propterminal-web",
            cpu=1024,
            memory_limit_mib=2048,
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.ARM64,
            ),
            volumes=[efs_volume_config],
        )

        # Use placeholder image until first ECR push via CI/CD.
        # After the first push, switch to:
        #   ecs.ContainerImage.from_ecr_repository(ecr_repo, tag="latest")
        web_container = web_task_def.add_container(
            "web",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repo, tag="latest"),
            port_mappings=[
                ecs.PortMapping(container_port=8000, protocol=ecs.Protocol.TCP),
            ],
            environment={
                "APP_MODE": "web",
                "EFS_MOUNT": "/data",
                "S3_BUCKET": s3_bucket.bucket_name,
                **shared_env,
            },
            secrets=secret_env,
            health_check=ecs.HealthCheck(
                command=["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
                interval=Duration.seconds(10),
                timeout=Duration.seconds(5),
                retries=3,
                start_period=Duration.seconds(30),
            ),
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="web",
                log_group=web_log_group,
            ),
        )

        web_container.add_mount_points(
            ecs.MountPoint(
                container_path="/data",
                source_volume="efs-data",
                read_only=False,
            ),
        )

        # Grant the web task EFS access
        efs_file_system.grant_root_access(web_task_def.task_role.grant_principal)

        # Grant S3 read/write
        s3_bucket.grant_read_write(web_task_def.task_role)

        # Grant DB secret read to web task execution role
        db_secret.grant_read(web_task_def.execution_role)

        # ------------------------------------------------------------------
        # Task Definition: propterminal-batch
        # ------------------------------------------------------------------
        batch_task_def = ecs.FargateTaskDefinition(
            self,
            "BatchTaskDef",
            family="propterminal-batch",
            cpu=512,
            memory_limit_mib=1024,
            runtime_platform=ecs.RuntimePlatform(
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
                cpu_architecture=ecs.CpuArchitecture.ARM64,
            ),
            volumes=[efs_volume_config],
        )

        # Same placeholder image — switch after first ECR push.
        batch_container = batch_task_def.add_container(
            "batch",
            image=ecs.ContainerImage.from_ecr_repository(ecr_repo, tag="latest"),
            environment={
                "APP_MODE": "batch",
                "EFS_MOUNT": "/data",
                "S3_BUCKET": s3_bucket.bucket_name,
                **shared_env,
            },
            secrets=secret_env,
            logging=ecs.LogDriver.aws_logs(
                stream_prefix="batch",
                log_group=batch_log_group,
            ),
        )

        batch_container.add_mount_points(
            ecs.MountPoint(
                container_path="/data",
                source_volume="efs-data",
                read_only=False,
            ),
        )

        efs_file_system.grant_root_access(batch_task_def.task_role.grant_principal)
        s3_bucket.grant_read_write(batch_task_def.task_role)

        # Grant DB secret read to batch task execution role
        db_secret.grant_read(batch_task_def.execution_role)

        # ------------------------------------------------------------------
        # ECS Fargate Service: propterminal-web (private subnets, NAT for egress)
        # ------------------------------------------------------------------
        web_service = ecs.FargateService(
            self,
            "WebService",
            service_name="propterminal-web",
            cluster=cluster,
            task_definition=web_task_def,
            desired_count=1,
            assign_public_ip=False,
            security_groups=[fargate_sg],
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
            ),
        )

        # Auto-scaling: scheduled scale up / scale down
        scaling = web_service.auto_scale_task_count(min_capacity=0, max_capacity=1)

        # Scale up at 4pm UTC (11am ET)
        scaling.scale_on_schedule(
            "ScaleUp",
            schedule=appscaling.Schedule.cron(hour="16", minute="0"),
            min_capacity=1,
            max_capacity=1,
        )

        # Scale down at 8am UTC (3am ET)
        scaling.scale_on_schedule(
            "ScaleDown",
            schedule=appscaling.Schedule.cron(hour="8", minute="0"),
            min_capacity=0,
            max_capacity=0,
        )

        # ------------------------------------------------------------------
        # Application Load Balancer
        # ------------------------------------------------------------------
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "PropTerminalAlb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC,
            ),
        )

        # Target group for the web service
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "WebTargetGroup",
            vpc=vpc,
            port=8000,
            targets=[web_service],
            health_check=elbv2.HealthCheck(
                path="/health",
                interval=Duration.seconds(30),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
                timeout=Duration.seconds(5),
            ),
        )

        if use_cognito and user_pool and user_pool_client and user_pool_domain:
            # --- Cognito ALB auth (Phase 2: subscription model) ---
            # HTTPS listener on port 443 (required for Cognito auth action)
            # Requires ACM certificate ARN via context ("acm_cert_arn")
            listener = alb.add_listener(
                "HttpsListener",
                port=443,
                open=True,
            )

            # /health bypasses auth (needed for ALB health checks)
            listener.add_action(
                "HealthBypass",
                priority=10,
                conditions=[elbv2.ListenerCondition.path_patterns(["/health"])],
                action=elbv2.ListenerAction.forward([target_group]),
            )

            # Default: Authenticate via Cognito, then forward
            listener.add_action(
                "CognitoAuth",
                action=elbv2_actions.AuthenticateCognitoAction(
                    user_pool=user_pool,
                    user_pool_client=user_pool_client,
                    user_pool_domain=user_pool_domain,
                    next=elbv2.ListenerAction.forward([target_group]),
                ),
            )

            # HTTP -> HTTPS redirect
            alb.add_listener(
                "HttpRedirect",
                port=80,
                open=True,
                action=elbv2.ListenerAction.redirect(
                    protocol="HTTPS",
                    port="443",
                    permanent=True,
                ),
            )
        else:
            # --- Password login only (Phase 1: MVP) ---
            # The app has its own login page with session-based auth.
            # Simple HTTP listener forwards all traffic; app handles auth.
            # TODO: Add HTTPS with ACM cert for production.
            listener = alb.add_listener(
                "HttpListener",
                port=80,
                open=True,
                default_action=elbv2.ListenerAction.forward([target_group]),
            )

        # ------------------------------------------------------------------
        # EventBridge Scheduled Tasks
        # ------------------------------------------------------------------
        # 1. Pre-game analysis — 4pm ET / 8pm UTC daily
        events.Rule(
            self,
            "PreGameAnalyze",
            rule_name="propterminal-pre-game-analyze",
            schedule=events.Schedule.expression("cron(0 20 * * ? *)"),
            targets=[
                events_targets.EcsTask(
                    cluster=cluster,
                    task_definition=batch_task_def,
                    task_count=1,
                    subnet_selection=ec2.SubnetSelection(
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                    security_groups=[fargate_sg],
                    assign_public_ip=False,
                    container_overrides=[
                        events_targets.ContainerOverride(
                            container_name=batch_container.container_name,
                            command=["analyze-all"],
                        ),
                    ],
                ),
            ],
        )

        # 2. Grade predictions — 2am ET / 7am UTC daily
        events.Rule(
            self,
            "GradePredictions",
            rule_name="propterminal-grade-predictions",
            schedule=events.Schedule.expression("cron(0 7 * * ? *)"),
            targets=[
                events_targets.EcsTask(
                    cluster=cluster,
                    task_definition=batch_task_def,
                    task_count=1,
                    subnet_selection=ec2.SubnetSelection(
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                    security_groups=[fargate_sg],
                    assign_public_ip=False,
                    container_overrides=[
                        events_targets.ContainerOverride(
                            container_name=batch_container.container_name,
                            command=["grade"],
                        ),
                    ],
                ),
            ],
        )

        # 3. Monthly retrain — 1st of month, 9am ET / 2pm UTC
        events.Rule(
            self,
            "RetrainMonthly",
            rule_name="propterminal-retrain-monthly",
            schedule=events.Schedule.expression("cron(0 14 1 * ? *)"),
            targets=[
                events_targets.EcsTask(
                    cluster=cluster,
                    task_definition=batch_task_def,
                    task_count=1,
                    subnet_selection=ec2.SubnetSelection(
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    ),
                    security_groups=[fargate_sg],
                    assign_public_ip=False,
                    container_overrides=[
                        events_targets.ContainerOverride(
                            container_name=batch_container.container_name,
                            command=["retrain"],
                        ),
                    ],
                ),
            ],
        )

        # ------------------------------------------------------------------
        # SNS Topic for Alerts
        # ------------------------------------------------------------------
        alerts_topic = sns.Topic(
            self,
            "AlertsTopic",
            topic_name="propterminal-alerts",
        )

        if alert_email:
            alerts_topic.add_subscription(
                sns_subs.EmailSubscription(alert_email),
            )

        # ------------------------------------------------------------------
        # CloudWatch Dashboard
        # ------------------------------------------------------------------
        dashboard = cw.Dashboard(
            self,
            "OpsDashboard",
            dashboard_name="propterminal-ops",
        )

        # ECS metrics
        ecs_cpu_metric = web_service.metric_cpu_utilization(
            period=Duration.minutes(5),
            statistic="Average",
        )
        ecs_memory_metric = web_service.metric_memory_utilization(
            period=Duration.minutes(5),
            statistic="Average",
        )

        # ALB metrics
        alb_request_count = cw.Metric(
            namespace="AWS/ApplicationELB",
            metric_name="RequestCount",
            dimensions_map={
                "LoadBalancer": alb.load_balancer_full_name,
            },
            period=Duration.minutes(5),
            statistic="Sum",
        )
        alb_response_time_p50 = cw.Metric(
            namespace="AWS/ApplicationELB",
            metric_name="TargetResponseTime",
            dimensions_map={
                "LoadBalancer": alb.load_balancer_full_name,
            },
            period=Duration.minutes(5),
            statistic="p50",
        )
        alb_response_time_p99 = cw.Metric(
            namespace="AWS/ApplicationELB",
            metric_name="TargetResponseTime",
            dimensions_map={
                "LoadBalancer": alb.load_balancer_full_name,
            },
            period=Duration.minutes(5),
            statistic="p99",
        )
        alb_5xx_count = cw.Metric(
            namespace="AWS/ApplicationELB",
            metric_name="HTTPCode_ELB_5XX_Count",
            dimensions_map={
                "LoadBalancer": alb.load_balancer_full_name,
            },
            period=Duration.minutes(5),
            statistic="Sum",
        )

        dashboard.add_widgets(
            cw.GraphWidget(
                title="ECS CPU Utilization",
                left=[ecs_cpu_metric],
                width=12,
            ),
            cw.GraphWidget(
                title="ECS Memory Utilization",
                left=[ecs_memory_metric],
                width=12,
            ),
        )
        dashboard.add_widgets(
            cw.GraphWidget(
                title="ALB Request Count",
                left=[alb_request_count],
                width=8,
            ),
            cw.GraphWidget(
                title="ALB Response Time (P50 / P99)",
                left=[alb_response_time_p50, alb_response_time_p99],
                width=8,
            ),
            cw.GraphWidget(
                title="ALB 5xx Errors",
                left=[alb_5xx_count],
                width=8,
            ),
        )

        # ------------------------------------------------------------------
        # CloudWatch Alarms
        # ------------------------------------------------------------------
        # Unhealthy host count alarm
        unhealthy_metric = cw.Metric(
            namespace="AWS/ApplicationELB",
            metric_name="UnHealthyHostCount",
            dimensions_map={
                "LoadBalancer": alb.load_balancer_full_name,
            },
            period=Duration.minutes(1),
            statistic="Maximum",
        )

        unhealthy_alarm = cw.Alarm(
            self,
            "WebUnhealthy",
            alarm_name="WebUnhealthy",
            metric=unhealthy_metric,
            threshold=0,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=5,
            datapoints_to_alarm=5,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="ALB has unhealthy hosts for 5+ minutes",
        )
        unhealthy_alarm.add_alarm_action(cw_actions.SnsAction(alerts_topic))

        # High 5xx error alarm
        high_5xx_alarm = cw.Alarm(
            self,
            "High5xx",
            alarm_name="High5xx",
            metric=alb_5xx_count,
            threshold=5,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_THRESHOLD,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
            alarm_description="ALB returned more than 5 5xx errors in 5 minutes",
        )
        high_5xx_alarm.add_alarm_action(cw_actions.SnsAction(alerts_topic))

        # Grading missed — placeholder using a custom metric
        # Publish this metric from the grading task on success; alarm fires
        # when metric is missing (i.e., the task did not run).
        grading_metric = cw.Metric(
            namespace="PropTerminal",
            metric_name="GradingCompleted",
            period=Duration.hours(24),
            statistic="Sum",
        )

        grading_alarm = cw.Alarm(
            self,
            "GradingMissed",
            alarm_name="GradingMissed",
            metric=grading_metric,
            threshold=1,
            comparison_operator=cw.ComparisonOperator.LESS_THAN_THRESHOLD,
            evaluation_periods=1,
            datapoints_to_alarm=1,
            treat_missing_data=cw.TreatMissingData.BREACHING,
            alarm_description=(
                "Grading task did not complete in the last 24 hours. "
                "Publish PropTerminal/GradingCompleted metric from the grade command."
            ),
        )
        grading_alarm.add_alarm_action(cw_actions.SnsAction(alerts_topic))

        # ------------------------------------------------------------------
        # Outputs
        # ------------------------------------------------------------------
        CfnOutput(self, "AlbDnsName", value=alb.load_balancer_dns_name)
        CfnOutput(self, "EcrRepositoryUri", value=ecr_repo.repository_uri)
        CfnOutput(self, "EcsClusterName", value=cluster.cluster_name)
        CfnOutput(self, "S3BucketName", value=s3_bucket.bucket_name)
        CfnOutput(self, "DbSecretArn", value=db_secret.secret_arn)
        CfnOutput(self, "RedisUrl",
                  value=f"redis://{redis_endpoint}:{redis_port}/0")

        if use_cognito and user_pool and user_pool_domain:
            CfnOutput(self, "CognitoUserPoolId", value=user_pool.user_pool_id)
            CfnOutput(self, "CognitoLoginUrl",
                      value=(
                          f"https://{user_pool_domain.domain_name}"
                          f".auth.{Stack.of(self).region}.amazoncognito.com/login"
                      ))
