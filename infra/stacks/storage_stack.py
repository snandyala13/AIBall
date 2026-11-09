import aws_cdk as cdk
from aws_cdk import (
    Aws,
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_efs as efs,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


class StorageStack(Stack):

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        vpc: ec2.IVpc,
        efs_sg: ec2.ISecurityGroup,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # EFS File System
        # ------------------------------------------------------------------
        self.efs_file_system = efs.FileSystem(
            self,
            "AiBallEfs",
            vpc=vpc,
            security_group=efs_sg,
            encrypted=True,
            throughput_mode=efs.ThroughputMode.BURSTING,
            lifecycle_policy=efs.LifecyclePolicy.AFTER_30_DAYS,
            removal_policy=RemovalPolicy.RETAIN,
            vpc_subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PUBLIC,
            ),
        )

        # Access point: /data owned by posix 1000:1000
        self.efs_access_point = self.efs_file_system.add_access_point(
            "DataAccessPoint",
            path="/data",
            posix_user=efs.PosixUser(uid="1000", gid="1000"),
            create_acl=efs.Acl(owner_uid="1000", owner_gid="1000", permissions="0755"),
        )

        # ------------------------------------------------------------------
        # S3 Bucket
        # ------------------------------------------------------------------
        self.s3_bucket = s3.Bucket(
            self,
            "AiBallArtifacts",
            bucket_name=f"aiball-artifacts-{Aws.ACCOUNT_ID}",
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    prefix="prediction-logs/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                            transition_after=Duration.days(90),
                        ),
                    ],
                    expiration=Duration.days(365),
                ),
            ],
        )

        # ------------------------------------------------------------------
        # Secrets Manager — placeholder secrets the user fills in later
        # ------------------------------------------------------------------
        self.secrets: dict[str, secretsmanager.Secret] = {}

        self.secrets["balldontlie"] = secretsmanager.Secret(
            self,
            "BallDontLieApiKey",
            secret_name="propterminal/balldontlie-api-key",
            secret_string_value=cdk.SecretValue.unsafe_plain_text(
                "PLACEHOLDER"
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.secrets["propline"] = secretsmanager.Secret(
            self,
            "PropLineApiKey",
            secret_name="propterminal/propline-api-key",
            secret_string_value=cdk.SecretValue.unsafe_plain_text(
                "PLACEHOLDER"
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.secrets["odds_api"] = secretsmanager.Secret(
            self,
            "OddsApiKey",
            secret_name="propterminal/odds-api-key",
            secret_string_value=cdk.SecretValue.unsafe_plain_text(
                "PLACEHOLDER"
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.secrets["ctg"] = secretsmanager.Secret(
            self,
            "CtgCredentials",
            secret_name="propterminal/ctg-credentials",
            secret_string_value=cdk.SecretValue.unsafe_plain_text(
                '{"email":"","password":""}'
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.secrets["pt_auth"] = secretsmanager.Secret(
            self,
            "PropTerminalAuth",
            secret_name="propterminal/auth",
            secret_string_value=cdk.SecretValue.unsafe_plain_text(
                '{"password_hash":"PLACEHOLDER","session_secret":"PLACEHOLDER"}'
            ),
            removal_policy=RemovalPolicy.DESTROY,
        )
